import torch, os, imageio, argparse
import torch.nn as nn
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
import time 
import pandas as pd
from diffsynth import WanVideoPipelineActEmbed, ModelManager, load_state_dict
from peft import LoraConfig, inject_adapter_in_model
import torchvision
from PIL import Image
import numpy as np
import h5py
import cv2


class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, max_num_frames=81, frame_interval=1, num_frames=81, height=480, width=832, is_i2v=False, samples_per_file=3, action_encoded_path=None):
        metadata = pd.read_csv(metadata_path)
        # Check if 'file_path' exists in metadata
        if 'file_path' in metadata.columns:
            self.path = metadata['file_path'].to_list()
            print(len(self.path))
        else:
            self.path = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]
        self.text = metadata["text"].to_list()
        self.file_name = metadata["file_name"].to_list()
        self.base_path = base_path
        self.action_encoded_path = action_encoded_path
        
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v
        self.samples_per_file = samples_per_file
            
        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

      
    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image


    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval:
            reader.close()
            return None
        
        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = np.array(frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        if self.is_i2v:
            return frames, first_frame
        else:
            return frames


    def load_video(self, file_path):
        start_frame_id = torch.randint(0, self.max_num_frames - (self.num_frames - 1) * self.frame_interval, (1,))[0]
        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process)
        return frames
    
    def load_hdf5(self, file_path):
        with h5py.File(file_path, 'r') as f:
            # Read the data from the specified field
            data = f['observations/images/cam_high']
            is_compressed = f.attrs.get('compress', False)
            total_frames = data.shape[0]  # Get the number of frames

            # Ensure the number of frames is sufficient
            if total_frames < self.num_frames:
                return None

            frames = []
            first_frame = None  # To store first frames if is_i2v is True          
            start_frame_id = torch.randint(0, total_frames - (self.num_frames - 1) * self.frame_interval, (1,)).item()
            end_frame_id = start_frame_id + self.num_frames * self.frame_interval

            frames = self.load_frames_from_hdf5(data, start_frame_id, end_frame_id, is_compressed=is_compressed)
            text = self.load_text_from_hdf5(file_path, start_frame_id, end_frame_id)
            actions = self.load_act_embed(file_path, start_frame_id, end_frame_id)
            print("shape of actions:", actions.shape)
            if self.is_i2v:
                frames, first_frame = frames
                return text, actions, frames, first_frame    
            else:
                return text, frames
            
    def load_act_embed(self, file_path, start_index, end_index):
        if self.action_encoded_path is not None:
            data = torch.load(self.action_encoded_path, weights_only=False)
            file_index = data['file_path'].index(file_path)
            action_data = data['encoded_action'][file_index]   
            action_data = action_data[start_index: end_index + 1: self.frame_interval]
            if not isinstance(action_data, np.ndarray):
                action_data = action_data.numpy()
            return action_data
        else:
            with h5py.File(file_path, 'r') as f:  
                action_data = f['action_embed'][start_index: end_index + 1: self.frame_interval]
                if action_data.ndim == 3 and action_data.shape[1] == 1:
                    action_data = action_data.squeeze(1) 
                return action_data
            
    def load_text_from_hdf5(self, file_path, start_frame_id, end_frame_id):
        with h5py.File(file_path, 'r') as f:
            # Check if 'substep_reasonings' exists and is not empty
            if 'substep_reasonings' in f and f['substep_reasonings'].shape[0] > end_frame_id:
                texts = [f['substep_reasonings'][i].decode('utf-8').strip() for i in range(start_frame_id, end_frame_id, self.frame_interval)]
                # Check if all texts are empty
                if all(text == "" for text in texts):
                    text = f['language_raw'][0]
                    text = text.decode('utf-8').strip()
                    return [text] 
                unique_texts = set(texts)
                if len(unique_texts) == 1:
                    return [texts[0]] 
                else:
                    return [" ".join(texts)]  
            # Fallback to 'language_raw' if 'substep_reasonings' doesn't exist
            text = f['language_raw'][0]
            text = text.decode('utf-8').strip()
        return [text]  # Wrap the text in a list

    def load_frames_from_hdf5(self, data, start_frame_id, end_frame_id, is_compressed=False, image_size=None):
        frames = data[start_frame_id: end_frame_id: self.frame_interval]
        frames = [Image.fromarray(frame) for frame in frames]
        
        # If frames are compressed, decompress them
        if is_compressed:
            frames = [cv2.imdecode(np.frombuffer(frame.tobytes(), np.uint8), cv2.IMREAD_COLOR) for frame in frames]
            if image_size is not None:
                frames = [cv2.resize(frame, eval(image_size)) for frame in frames]
            frames = [Image.fromarray(frame) for frame in frames]

        # Crop and resize frames
        frames = [self.crop_and_resize(frame) for frame in frames]
        
        # Capture the first frame after crop and resize
        first_frame = np.array(frames[0])
        
        frames = [self.frame_process(frame) for frame in frames]
        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        if self.is_i2v:
            return frames, first_frame
        else:
            return frames

    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False

    def is_hdf5(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in ["h5", "hdf5"]
    
    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        first_frame = frame
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame


    def __getitem__(self, data_id):
        actual_data_id = data_id // self.samples_per_file  
        text = self.text[actual_data_id]
        path = self.path[actual_data_id]
        to_path = path

        if self.is_image(path):
            if self.is_i2v:
                raise ValueError(f"{path} is not a video. I2V model doesn't support image-to-image training.")
            video = self.load_image(path)
        elif self.is_hdf5(path):
            same_file_sample_id = data_id % self.samples_per_file
            from_path = self.path[actual_data_id]
            text, actions, video, first_frame = self.load_hdf5(from_path)
            parent_dir_name = os.path.basename(os.path.dirname(from_path))
            to_path = os.path.join(self.base_path, "train", f"{parent_dir_name}_{self.file_name[actual_data_id]}_{same_file_sample_id}")
        else:
            # mp4
            video, first_frame = self.load_video(path)

        if self.is_i2v:
            data = {"text": text, "video": video, "path": to_path, "first_frame": first_frame, "actions": actions}
        else:
            data = {"text": text, "video": video, "path": to_path}
        return data
    

    def __len__(self):
        return len(self.path)*self.samples_per_file



class LightningModelForDataProcess(pl.LightningModule):
    def __init__(self, text_encoder_path, vae_path, image_encoder_path=None, tiled=False, tile_size=(34, 34), tile_stride=(18, 16), encode_mode="dexvla", action_alpha=0.1, action_dim=1280):
        super().__init__()
        model_path = [text_encoder_path, vae_path]
        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu", custom_params={"action_alpha": action_alpha, "action_dim": action_dim})
        model_manager.load_models(model_path)
        self.pipe = WanVideoPipelineActEmbed.from_model_manager(model_manager)

        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        
    def test_step(self, batch, batch_idx):
        text, video, path, actions = batch["text"][0], batch["video"], batch["path"][0], batch["actions"]
        
        self.pipe.device = self.device
        if video is not None:
            # prompt
            prompt_emb = self.pipe.encode_prompt(text)
            # video
            video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            latents = self.pipe.encode_video(video, **self.tiler_kwargs)[0]
            # image
            if "first_frame" in batch:
                first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())
                _, _, num_frames, height, width = video.shape
                image_emb = self.pipe.encode_image(first_frame, num_frames, height, width)
            else:
                image_emb = {}
            data = {"latents": latents, "prompt_emb": prompt_emb, "image_emb": image_emb, "action_emb": actions}
            torch.save(data, path + ".tensors.pth")   



class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, steps_per_epoch):
        # read all .tensors.pth files from the train directory
        self.path = [os.path.join(base_path, "train", file_name) for file_name in os.listdir(os.path.join(base_path, "train")) if file_name.endswith(".tensors.pth")]
        print(len(self.path), "tensors cached in metadata.")
        assert len(self.path) > 0
        
        self.steps_per_epoch = steps_per_epoch


    def __getitem__(self, index):
        data_id = torch.randint(0, len(self.path), (1,))[0]
        data_id = (data_id + index) % len(self.path)  # For fixed seed.
        path = self.path[data_id]
        
        data = torch.load(path, weights_only=True, map_location="cpu")
           
        return data
    

    def __len__(self):
        return self.steps_per_epoch



class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        learning_rate=1e-5,
        lora_rank=4, lora_alpha=4, train_architecture="lora", lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming",
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        pretrained_lora_path=None,
        encode_mode="dexvla",
        action_alpha=0.1,
        action_dim=1280
    ):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu", custom_params={"action_alpha": action_alpha, "action_dim": action_dim})
        self.encode_mode = encode_mode
        if os.path.isfile(dit_path):
            model_manager.load_models([dit_path])
        else:
            dit_path = dit_path.split(",")
            model_manager.load_models([dit_path])
        

        self.pipe = WanVideoPipelineActEmbed.from_model_manager(model_manager)

        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.freeze_parameters()
        if train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.denoising_model(),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
            for layer in self.pipe.denoising_model().action_proj:
                if isinstance(layer, nn.Linear):
                    layer.weight.requires_grad = True
                    if layer.bias is not None:
                        layer.bias.requires_grad = True
            self.pipe.denoising_model().action_alpha.requires_grad = True
            trainable_params = [name for name, param in self.pipe.denoising_model().named_parameters() if param.requires_grad]
            # print("Trainable parameters:", trainable_params)
        else:
            self.pipe.denoising_model().requires_grad_(True)
        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        
        
    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()
        
        
    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None):
        # Add LoRA to UNet
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True
            
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)

        for name, param in model.named_parameters():
            if "action_proj" in name and ("weight" in name or "bias" in name):
                param.requires_grad = True  

        for param in model.parameters():
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)
                
        # Lora pretrained lora weights
        if pretrained_lora_path is not None:
            state_dict = load_state_dict(pretrained_lora_path)
            # filter not-exist keys
            state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and not k.startswith("action_proj") or  k  in model.state_dict()}
            
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")
    

    def training_step(self, batch, batch_idx):
        # Add this line to print the keys in the batch

        # Data
        latents = batch["latents"].to(self.device)
        prompt_emb = batch["prompt_emb"]
        prompt_emb["context"] = prompt_emb["context"][0].to(self.device)
        image_emb = batch["image_emb"]
        # print("action_emb.shape:", action_emb.shape)
        action_emb = batch["action_emb"]
        
        if self.encode_mode == "onehot":
            # Flatten the last two dimensions of action_emb
            action_emb = action_emb.view(action_emb.size(0), action_emb.size(1), action_emb.size(2), -1)

        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = image_emb["clip_feature"][0].to(self.device)
        if "y" in image_emb:
            image_emb["y"] = image_emb["y"][0].to(self.device)

        # Loss
        self.pipe.device = self.device
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        # Compute loss
        noise_pred = self.pipe.denoising_model()(
            noisy_latents, timestep=timestep, **prompt_emb, **extra_input, **image_emb, action=action_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
        )
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)

        # Record log
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss


    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters()))
        # print("Trainable parameter names:", trainable_param_names)  # Print trainable parameter names
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        # print("Trainable parameter names set:", trainable_param_names)  # Print trainable parameter names set
        state_dict = self.pipe.denoising_model().state_dict()
        lora_state_dict = {}
        for name, param in state_dict.items():
            if name in trainable_param_names:
                if torch.isnan(param).any():
                    print(f"Warning: NaN values found in parameter {name}")
                lora_state_dict[name] = param
                print(f"Added {name} to lora_state_dict")  # Print each parameter added to lora_state_dict
        checkpoint.update(lora_state_dict)



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--task",
        type=str,
        default="data_process",
        required=True,
        choices=["data_process", "train"],
        help="Task. `data_process` or `train`.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        help="Path of image encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default=None,
        help="Path of DiT.",
    )
    parser.add_argument(
        "--tiled",
        default=False,
        action="store_true",
        help="Whether enable tile encode in VAE. This option can reduce VRAM required.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size (height) in VAE.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size (width) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride (height) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride (width) in VAE.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q,k,v,o,ffn.0,ffn.2",
        help="Layers with LoRA modules.",
    )
    parser.add_argument(
        "--init_lora_weights",
        type=str,
        default="kaiming",
        choices=["gaussian", "kaiming"],
        help="The initializing method of LoRA weight.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=4.0,
        help="The weight of the LoRA update matrices.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument(
        "--train_architecture",
        type=str,
        default="lora",
        choices=["lora", "full"],
        help="Model structure to train. LoRA training or full training.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        help="Pretrained LoRA path. Required if the training is resumed.",
    )
    parser.add_argument(
        "--use_swanlab",
        default=False,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode",
        default=None,
        help="SwanLab mode (cloud or local).",
    )
    parser.add_argument(
        "--samples_per_file",
        type=int,
        default=3,
        help="Number of samples per file.",
    )
    parser.add_argument(
        "--action_encoded_path",
        type=str,
        default=None,
        help="Path of onehot encoded action data.",
    )
    parser.add_argument(
        "--encode_mode",
        type=str,
        default="dexvla",
        choices=["onehot", "dexvla", "vqvae", 'pi0', 'dp', 'openvla'],
        help="The mode of the action data.",
    )
    parser.add_argument(
        "--action_alpha",
        type=float,
        default=0.1,
        help="The alpha of the action data.",
    )
    parser.add_argument(
        "--action_dim",
        type=int,
        default=1280,
        help="The dimension of the action data.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="The version of the model.",
    )
    
    args = parser.parse_args()
    return args


def data_process(args):
    dataset = TextVideoDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, "metadata.csv"),
        max_num_frames=args.num_frames,
        frame_interval=1,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        is_i2v=args.image_encoder_path is not None,
        samples_per_file=args.samples_per_file,
        action_encoded_path=args.action_encoded_path
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )
    model = LightningModelForDataProcess(
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        vae_path=args.vae_path,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
        encode_mode=args.encode_mode,
        action_alpha=args.action_alpha,
        action_dim=args.action_dim
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        default_root_dir=args.output_path,
    )
    trainer.test(model, dataloader)
    
    
def train(args):
    dataset = TensorDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, "metadata.csv"),
        steps_per_epoch=args.steps_per_epoch
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )
    model = LightningModelForTrain(
        dit_path=args.dit_path,
        learning_rate=args.learning_rate,
        train_architecture=args.train_architecture,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        init_lora_weights=args.init_lora_weights,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        pretrained_lora_path=args.pretrained_lora_path,
        encode_mode=args.encode_mode,
        action_alpha=args.action_alpha,
        action_dim=args.action_dim
    )
    if args.use_swanlab:
        from swanlab.integration.pytorch_lightning import SwanLabLogger
        swanlab_config = {"UPPERFRAMEWORK": "DiffSynth-Studio"}
        swanlab_config.update(vars(args))
        swanlab_logger = SwanLabLogger(
            project="wan", 
            name="wan",
            config=swanlab_config,
            mode=args.swanlab_mode,
            logdir=os.path.join(args.output_path, "swanlog"),
        )
        logger = [swanlab_logger]
    else:
        version = args.version if args.version is not None else str(int(time.time()))
        logger = TensorBoardLogger(save_dir=args.output_path, version=version)
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1)]
    )
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    args = parse_args()
    if args.task == "data_process":
        data_process(args)
    elif args.task == "train":
        train(args)
