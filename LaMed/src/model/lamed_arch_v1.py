from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_mm_projector
from .segmentation_module.builder import build_segmentation_module
from LaMed.src.model.loss import BCELoss, BinaryDiceLoss


class AttentionAggregator(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        print(f"AttentionAggregator hidden_size: {hidden_size}, num_heads: {num_heads}")
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first = True )
        self.layer_norm = nn.LayerNorm(hidden_size)
        print(f"out_proj_weight shape: {self.attention.out_proj.weight.shape}")
        # Linear layer to reduce sequence length from num_parts * 256 to 256
        total_seq_length = 8 * 256  # Define num_parts appropriately
        self.reduce_seq_length = nn.Linear(total_seq_length, 256)

    def forward(self, image_embeddings):
        # image_embeddings shape: [batch_size, seq_length, hidden_size]
        # print(f"Initial image_embeddings shape: {image_embeddings.shape} and type: {type(image_embeddings)}")
        # image_embeddings = image_embeddings.as_tensor()
        # print(f"Initial image_embeddings shape CONVERTED: {image_embeddings.shape} and type: {type(image_embeddings)}")
        # Debug projection weights and bias
        # print(f"out_proj_weight shape: {self.attention.out_proj.weight.shape}")
        # print(f"out_proj_bias shape: {self.attention.out_proj.bias.shape}")
        # self.attention.out_proj.bias = nn.Parameter(torch.zeros(2048))
        # self.attention.out_proj.weight = nn.Parameter(torch.ones(2048, 2048))
        
        # Transpose for MultiheadAttention: [seq_length, batch_size, hidden_size]
        # image_embeddings = image_embeddings.transpose(0, 1)
        # print(f"image_embeddings shape after transpose: {image_embeddings.shape}")
        # Apply multihead attention
        # attn_output, _ = self.attention(image_embeddings, image_embeddings, image_embeddings)
        attn_output, _ = self.attention(image_embeddings, image_embeddings, image_embeddings , need_weights=False)
        
        # print(f"attn_output shape after attention: {attn_output}")

        # Apply layer normalization
        attn_output = self.layer_norm(attn_output)

        # # Transpose back: [batch_size, seq_length, hidden_size]
        # attn_output = attn_output.transpose(0, 1)

        # Reduce sequence length to 256
        attn_output = attn_output.transpose(1, 2)  # Shape: [batch_size, hidden_size, seq_length]
        aggregated_features = self.reduce_seq_length(attn_output)  # Shape: [batch_size, hidden_size, 256]
        aggregated_features = aggregated_features.transpose(1, 2)  # Shape: [batch_size, 256, hidden_size]

        return aggregated_features

class LamedMetaModel:
    def __init__(self, config):
        super(LamedMetaModel, self).__init__(config)

        self.config = config
        self.seg_enable = False

        if hasattr(config, "vision_tower"):
            self.vision_tower = build_vision_tower(config)
            self.mm_projector = build_mm_projector(config)

        if hasattr(config, "segmentation_module") and config.segmentation_module is not None:
            self.seg_enable = True
            self.seg_module = build_segmentation_module(config)

            self.seg_projector = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(config.hidden_size, config.mm_hidden_size),
                nn.Dropout(0.1),
            )

            self.dice_loss = BinaryDiceLoss()
            self.bce_loss = BCELoss()
        
        # Initialize Attention Aggregator
        if hasattr(config, "attention_aggregator"):
            print(f" in HASATTR AttentionAggregator hidden_size: {config.hidden_size}, num_heads: {config.num_heads}")
            # self.attention_aggregator = AttentionAggregator(config.hidden_size, config.num_heads)
            self.attention_aggregator.requires_grad_(True) # Enable training the attention aggregator

    def initialize_attention_aggregator(self, model_args):
        self.config.num_heads = model_args.num_heads
        print(f"IN initialize AttentionAggregator hidden_size: {self.config.hidden_size}, num_heads: {self.config.num_heads}")
        if self.get_attention_aggregator() is None:
            self.attention_aggregator = AttentionAggregator(self.config.hidden_size, self.config.num_heads)


    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        return vision_tower
    def get_attention_aggregator(self):
        attention_aggregator = getattr(self, 'attention_aggregator', None)
        return attention_aggregator
    def initialize_vision_modules(self, model_args):
        self.config.image_channel = model_args.image_channel
        self.config.image_size = model_args.image_size
        self.config.patch_size = model_args.patch_size

        self.config.vision_tower = model_args.vision_tower
        self.config.vision_select_layer = model_args.vision_select_layer
        self.config.vision_select_feature = model_args.vision_select_feature

        self.config.mm_projector_type = model_args.mm_projector_type
        self.config.proj_layer_type = model_args.proj_layer_type
        self.config.proj_layer_num = model_args.proj_layer_num
        self.config.proj_pooling_type = model_args.proj_pooling_type
        self.config.proj_pooling_size = model_args.proj_pooling_size

        # vision tower
        if self.get_vision_tower() is None:
            self.vision_tower = build_vision_tower(self.config)
            # If you have a more robust vision encoder, try freezing the vision tower by requires_grad_(False)
            self.vision_tower.requires_grad_(not model_args.freeze_vision_tower)


        if model_args.pretrain_vision_model is not None:
            vision_model_weights = torch.load(model_args.pretrain_vision_model, map_location='cpu')
            self.vision_tower.vision_tower.load_state_dict(vision_model_weights, strict=True)

        self.config.mm_hidden_size = self.vision_tower.hidden_size

        # mm_projector
        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_mm_projector(self.config)

        if model_args.pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=True)

    def initialize_seg_modules(self, model_args):
        self.config.segmentation_module = model_args.segmentation_module

        # segmentation_module
        if getattr(self, 'seg_module', None) is None:
            self.seg_module = build_segmentation_module(self.config)
            self.seg_projector = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.config.hidden_size, self.config.mm_hidden_size),
                nn.Dropout(0.1),
            )
            self.seg_enable = True

        if model_args.pretrain_seg_module is not None:
            seg_module_weights = torch.load(model_args.pretrain_seg_module, map_location='cpu')
            new_state_dict = {}
            for key, value in seg_module_weights.items():
                if key.startswith('model.text_encoder.') or key.startswith('text_encoder.'):
                    continue
                if key.startswith('model.'):
                    new_key = key[len('model.'):]
                    new_state_dict[new_key] = value
            self.seg_module.load_state_dict(new_state_dict, strict=True)

        self.dice_loss = BinaryDiceLoss()
        self.bce_loss = BCELoss()

class LamedMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features
    def encode_images2(self, images):
        # images shape: [batch_size, num_parts, channels, D, H, W]
        # print(f"Input image shape: {images.shape}")
        batch_size, num_parts, channels, D, H, W = images.shape

        # Reshape to process each part individually
        images = images.view(batch_size * num_parts, channels, D, H, W)
        # ViT output shape: [batch_size * num_parts, tokens, embedding_dim]
        image_features = self.get_model().get_vision_tower()(images)
        # print(f"ViT output shape: {image_features.shape}")
        # Apply mm_projector to each part
        # mm_projector output shape: [batch_size * num_parts, 256, 3072]
        image_features = self.get_model().mm_projector(image_features)
        # print(f"mm_projector output shape: {image_features.shape}")
        # Reshape back to [batch_size, num_parts, 256, 3072]
        image_features = image_features.view(batch_size, num_parts, 256, 3072)

        # Combine parts by concatenating along the sequence length
        # Resulting shape: [batch_size, num_parts * 256, 3072]
        image_embeddings = image_features.contiguous().view(batch_size, num_parts * 256, 3072)
        # print(f"Input to attention : {image_embeddings}")
        # print(f"Input to attention shape: {image_embeddings.shape}")
        # Apply attention aggregator to reduce sequence length back to 256
        # Final output shape: [batch_size, 256, 3072]
        # print(f"Attention aggregator weights shape: {self.get_model().get_attention_aggregator().attention.out_proj.weight.shape}")
        image_features = self.get_model().get_attention_aggregator()(image_embeddings)
        # print(f"Output of attention aggregator shape: {image_features.shape}")
        return image_features
    def prepare_inputs_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images,
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        else:
            image_features = self.encode_images2(images)
            # image_features = []
            # # Get image features for each image part
            # for i in range(images.shape[1]):
            #     image_part = images[:, i, :, :, :]
            #     print(f"Image part shape: {image_part.shape}")
            #     image_feature = self.encode_images(image_part)
            #     image_features.append(image_feature)

            # # Stack image features to form a tensor
            # image_features = torch.stack(image_features)
            # # Transpose to [batch_size, num_parts, tokens, embedding_dim]
            # image_features = image_features.transpose(0, 1)
            # print(f"Image features shape: {image_features.shape}")
            # # Aggregate image features using attention
            # batch_size, num_parts, tokens, embedding_dim = image_features.shape
            # image_embeddings = image_features.reshape(batch_size, num_parts * tokens, embedding_dim)
            # print(f"Image embeddings shape: {image_embeddings.shape}")
            # image_features = self.get_model().attention_aggregator(image_embeddings)
            # print(f"Aggregated image features shape: {image_features.shape}")
            # image_features = image_features.mean(dim=0)  # [1, mm_hidden_size]
            # print(f"Meaned image features shape: {image_features.shape}")
            inputs_embeds = self.get_model().embed_tokens(input_ids)
            inputs_embeds = torch.cat(
                (inputs_embeds[:, :1, :], image_features, inputs_embeds[:, (image_features.shape[1] + 1):, :]), dim=1)
        return None, position_ids, attention_mask, past_key_values, inputs_embeds, labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        num_new_tokens = model_args.num_new_tokens

        self.resize_token_embeddings(len(tokenizer))

        if num_new_tokens > 0:
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
            else:
                # we add 4 new tokens
                # if new tokens need input, please train input_embeddings
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                # if new tokens need predict, please train output_embeddings
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = True

        if model_args.pretrain_mm_mlp_adapter:
            mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
            embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']

            if input_embeddings.shape == embed_tokens_weight.shape:
                input_embeddings = embed_tokens_weight
            elif embed_tokens_weight.shape[0] == num_new_tokens:
                input_embeddings[-num_new_tokens:] = embed_tokens_weight
            else:
                raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")