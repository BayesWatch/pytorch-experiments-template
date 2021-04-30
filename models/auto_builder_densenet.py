from __future__ import print_function

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print

from models import ClassificationModel
from models import Conv2dBNLeakyReLU


class DenseBlock(nn.Module):
    def __init__(
        self,
        num_filters,
        dilation_factor,
        kernel_size,
        stride,
        downsample_output_size=None,
    ):
        super(DenseBlock, self).__init__()
        self.num_filters = num_filters
        self.dilation_factor = dilation_factor
        self.downsample_output_size = downsample_output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.is_built = False

    def build(self, input_shape):
        dummy_x = torch.zeros(input_shape)

        self.conv_bn_relu_in = Conv2dBNLeakyReLU(
            out_channels=self.num_filters,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.dilation_factor,
            dilation=self.dilation_factor,
            bias=False,
        )

        out = self.conv_bn_relu_in.forward(dummy_x)

        if self.downsample_output_size is not None:
            out = F.adaptive_avg_pool2d(
                output_size=self.downsample_output_size, input=out
            )
            dummy_x = F.adaptive_avg_pool2d(
                output_size=self.downsample_output_size, input=dummy_x
            )

        out = torch.cat([dummy_x, out], dim=1)

        self.is_built = True
        print(
            "Built module",
            self.__class__.__name__,
            "with input -> output sizes",
            dummy_x.shape,
            out.shape,
        )

    def forward(self, x):

        if not self.is_built:
            self.build(x.shape)

        out = self.conv_bn_relu_in.forward(x)

        if self.downsample_output_size is not None:
            out = F.adaptive_avg_pool2d(
                output_size=self.downsample_output_size, input=out
            )
            x = F.adaptive_avg_pool2d(output_size=self.downsample_output_size, input=x)

        out = torch.cat([x, out], dim=1)

        return out


class DenseNetEmbedding(nn.Module):
    def __init__(self, num_filters, num_stages, num_blocks, dilated=False):
        super(DenseNetEmbedding, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.is_built = False
        self.num_filters = num_filters
        self.num_stages = num_stages
        self.num_blocks = num_blocks
        self.dilated = dilated

    def build(self, input_shape):
        dummy_x = torch.zeros(input_shape)
        out = dummy_x

        dilation_factor = 1

        self.layer_dict["stem_conv"] = DenseBlock(
            num_filters=self.num_filters * 4,
            dilation_factor=1,
            stride=1,
            kernel_size=3,
            downsample_output_size=int(np.floor(out.shape[-1] / 2)),
        )

        out = self.layer_dict["stem_conv"].forward(out)

        for stage_idx in range(self.num_stages):

            for block_idx in range(self.num_blocks):

                if self.dilated:
                    dilation_factor = 2 ** block_idx

                self.layer_dict[
                    "stage_{}_block_{}".format(stage_idx, block_idx)
                ] = DenseBlock(
                    num_filters=self.num_filters,
                    dilation_factor=dilation_factor,
                    downsample_output_size=None,
                    stride=1,
                    kernel_size=3,
                )

                out = self.layer_dict[
                    "stage_{}_block_{}".format(stage_idx, block_idx)
                ].forward(out)

            self.layer_dict["stage_{}_dim_reduction".format(stage_idx)] = DenseBlock(
                num_filters=self.num_filters,
                dilation_factor=1,
                stride=1,
                kernel_size=3,
                downsample_output_size=int(np.floor(out.shape[-1] / 2)),
            )

            out = self.layer_dict["stage_{}_dim_reduction".format(stage_idx)].forward(
                out
            )

        self.is_built = True
        print(
            "Built module",
            self.__class__.__name__,
            "with input -> output sizes",
            dummy_x.shape,
            out.shape,
        )

    def forward(self, x):

        if not self.is_built:
            self.build(input_shape=x.shape)

        out = x

        out = self.layer_dict["stem_conv"].forward(out)

        for stage_idx in range(self.num_stages):

            for block_idx in range(self.num_blocks):

                out = self.layer_dict[
                    "stage_{}_block_{}".format(stage_idx, block_idx)
                ].forward(out)

            out = self.layer_dict["stage_{}_dim_reduction".format(stage_idx)].forward(
                out
            )

        return out


class AutoDenseNet(ClassificationModel):
    def __init__(
        self, num_classes, num_filters, num_stages, num_blocks, dilated=False, **kwargs
    ):
        feature_embedding_modules = [DenseNetEmbedding]
        feature_embeddings_args = [
            dict(
                num_filters=num_filters,
                num_stages=num_stages,
                num_blocks=num_blocks,
                dilated=dilated,
            )
        ]

        super(AutoDenseNet, self).__init__(
            num_classes=num_classes,
            feature_embedding_module_list=feature_embedding_modules,
            feature_embedding_args=feature_embeddings_args,
        )
