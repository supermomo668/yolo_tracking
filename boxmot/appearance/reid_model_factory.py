import sys
import time
from collections import OrderedDict

import torch

from boxmot.utils import logger as LOGGER

__model_types = [
    "resnet50",
    "resnet101",
    "mlfn",
    "hacnn",
    "mobilenetv2_x1_0",
    "mobilenetv2_x1_4",
    "osnet_x1_0",
    "osnet_x0_75",
    "osnet_x0_5",
    "osnet_x0_25",
    "osnet_ibn_x1_0",
    "osnet_ain_x1_0",
    "lmbn_n",
    "clip",
]

lmbn_loc = 'https://github.com/mikel-brostrom/yolov8_tracking/releases/download/v9.0/'

__trained_urls = {
    # resnet50
    "resnet50_market1501.pt": "https://drive.google.com/uc?id=1dUUZ4rHDWohmsQXCRe2C_HbYkzz94iBV",
    "resnet50_dukemtmcreid.pt": "https://drive.google.com/uc?id=17ymnLglnc64NRvGOitY3BqMRS9UWd1wg",
    "resnet50_msmt17.pt": "https://drive.google.com/uc?id=1ep7RypVDOthCRIAqDnn4_N-UhkkFHJsj",
    "resnet50_fc512_market1501.pt": "https://drive.google.com/uc?id=1kv8l5laX_YCdIGVCetjlNdzKIA3NvsSt",
    "resnet50_fc512_dukemtmcreid.pt": "https://drive.google.com/uc?id=13QN8Mp3XH81GK4BPGXobKHKyTGH50Rtx",
    "resnet50_fc512_msmt17.pt": "https://drive.google.com/uc?id=1fDJLcz4O5wxNSUvImIIjoaIF9u1Rwaud",
    # mlfn
    "mlfn_market1501.pt": "https://drive.google.com/uc?id=1wXcvhA_b1kpDfrt9s2Pma-MHxtj9pmvS",
    "mlfn_dukemtmcreid.pt": "https://drive.google.com/uc?id=1rExgrTNb0VCIcOnXfMsbwSUW1h2L1Bum",
    "mlfn_msmt17.pt": "https://drive.google.com/uc?id=18JzsZlJb3Wm7irCbZbZ07TN4IFKvR6p-",
    # hacnn
    "hacnn_market1501.pt": "https://drive.google.com/uc?id=1LRKIQduThwGxMDQMiVkTScBwR7WidmYF",
    "hacnn_dukemtmcreid.pt": "https://drive.google.com/uc?id=1zNm6tP4ozFUCUQ7Sv1Z98EAJWXJEhtYH",
    "hacnn_msmt17.pt": "https://drive.google.com/uc?id=1MsKRtPM5WJ3_Tk2xC0aGOO7pM3VaFDNZ",
    # mobilenetv2
    "mobilenetv2_x1_0_market1501.pt": "https://drive.google.com/uc?id=18DgHC2ZJkjekVoqBWszD8_Xiikz-fewp",
    "mobilenetv2_x1_0_dukemtmcreid.pt": "https://drive.google.com/uc?id=1q1WU2FETRJ3BXcpVtfJUuqq4z3psetds",
    "mobilenetv2_x1_0_msmt17.pt": "https://drive.google.com/uc?id=1j50Hv14NOUAg7ZeB3frzfX-WYLi7SrhZ",
    "mobilenetv2_x1_4_market1501.pt": "https://drive.google.com/uc?id=1t6JCqphJG-fwwPVkRLmGGyEBhGOf2GO5",
    "mobilenetv2_x1_4_dukemtmcreid.pt": "https://drive.google.com/uc?id=12uD5FeVqLg9-AFDju2L7SQxjmPb4zpBN",
    "mobilenetv2_x1_4_msmt17.pt": "https://drive.google.com/uc?id=1ZY5P2Zgm-3RbDpbXM0kIBMPvspeNIbXz",
    # osnet
    "osnet_x1_0_market1501.pt": "https://drive.google.com/uc?id=1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA",
    "osnet_x1_0_dukemtmcreid.pt": "https://drive.google.com/uc?id=1QZO_4sNf4hdOKKKzKc-TZU9WW1v6zQbq",
    "osnet_x1_0_msmt17.pt": "https://drive.google.com/uc?id=112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M",
    "osnet_x0_75_market1501.pt": "https://drive.google.com/uc?id=1ozRaDSQw_EQ8_93OUmjDbvLXw9TnfPer",
    "osnet_x0_75_dukemtmcreid.pt": "https://drive.google.com/uc?id=1IE3KRaTPp4OUa6PGTFL_d5_KQSJbP0Or",
    "osnet_x0_75_msmt17.pt": "https://drive.google.com/uc?id=1QEGO6WnJ-BmUzVPd3q9NoaO_GsPNlmWc",
    "osnet_x0_5_market1501.pt": "https://drive.google.com/uc?id=1PLB9rgqrUM7blWrg4QlprCuPT7ILYGKT",
    "osnet_x0_5_dukemtmcreid.pt": "https://drive.google.com/uc?id=1KoUVqmiST175hnkALg9XuTi1oYpqcyTu",
    "osnet_x0_5_msmt17.pt": "https://drive.google.com/uc?id=1UT3AxIaDvS2PdxzZmbkLmjtiqq7AIKCv",
    "osnet_x0_25_market1501.pt": "https://drive.google.com/uc?id=1z1UghYvOTtjx7kEoRfmqSMu-z62J6MAj",
    "osnet_x0_25_dukemtmcreid.pt": "https://drive.google.com/uc?id=1eumrtiXT4NOspjyEV4j8cHmlOaaCGk5l",
    "osnet_x0_25_msmt17.pt": "https://drive.google.com/uc?id=1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF",
    # osnet_ain | osnet_ibn
    "osnet_ibn_x1_0_msmt17.pt": "https://drive.google.com/uc?id=1q3Sj2ii34NlfxA4LvmHdWO_75NDRmECJ",
    "osnet_ain_x1_0_msmt17.pt": "https://drive.google.com/uc?id=1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal",
    # lmbn
    "lmbn_n_duke.pt": lmbn_loc + "lmbn_n_duke.pth",
    "lmbn_n_market.pt": lmbn_loc + "lmbn_n_market.pth",
    "lmbn_n_cuhk03_d.pt": lmbn_loc + "lmbn_n_cuhk03_d.pth",
    # clip
    "clip_market1501.pt": "https://drive.google.com/uc?id=1GnyAVeNOg3Yug1KBBWMKKbT2x43O5Ch7",
    "clip_duke.pt": "https://drive.google.com/uc?id=1ldjSkj-7pXAWmx8on5x0EftlCaolU4dY",
    "clip_veri.pt": "https://drive.google.com/uc?id=1RyfHdOBI2pan_wIGSim5-l6cM4S2WN8e",
    "clip_vehicleid.pt": "https://drive.google.com/uc?id=1RyfHdOBI2pan_wIGSim5-l6cM4S2WN8e"
}


def show_downloadable_models():
    LOGGER.info("\nAvailable .pt ReID models for automatic download")
    LOGGER.info(list(__trained_urls.keys()))


def get_model_url(model):
    if model.name in __trained_urls:
        return __trained_urls[model.name]
    else:
        None


def is_model_in_model_types(model):
    if model.name in __model_types:
        return True
    else:
        return False


def get_model_name(model):
    for x in __model_types:
        if x in model.name:
            return x
    return None


def download_url(url, dst):
    """Downloads file from a url to a destination.

    Args:
        url (str): url to download file.
        dst (str): destination path.
    """
    from six.moves import urllib

    LOGGER.info('* url="{}"'.format(url))
    LOGGER.info('* destination="{}"'.format(dst))

    def _reporthook(count, block_size, total_size):
        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(
            "\r...%d%%, %d MB, %d KB/s, %d seconds passed"
            % (percent, progress_size / (1024 * 1024), speed, duration)
        )
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dst, _reporthook)
    sys.stdout.write("\n")


def load_pretrained_weights(model, weight_path):
    r"""Loads pretrianed weights to model.

    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".

    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.

    Examples::
        >>> from boxmot.appearance.backbones import build_model
        >>> from boxmot.appearance.reid_model_factory import load_pretrained_weights
        >>> weight_path = 'log/my_model/model-best.pth.tar'
        >>> model = build_model()
        >>> load_pretrained_weights(model, weight_path)
    """

    if not torch.cuda.is_available():
        checkpoint = torch.load(weight_path, map_location=torch.device("cpu"))
    else:
        checkpoint = torch.load(weight_path)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()

    if "lmbn" in str(weight_path):
        model.load_state_dict(model_dict, strict=True)
    elif "clip" in str(weight_path):
        def forward_override(self, x: torch.Tensor, cv_emb=None, old_forward=None):
            _, image_features, image_features_proj = old_forward(x, cv_emb)
            return torch.cat([image_features[:, 0], image_features_proj[:, 0]], dim=1)
        # print('model.load_param(str(weight_path))', str(weight_path))
        model.load_param(str(weight_path))
        model = model.image_encoder
        # old_forward = model.forward
        # model.forward = lambda *args, **kwargs: forward_override(model, old_forward=old_forward, *args, **kwargs)
        LOGGER.success(
            f'Successfully loaded pretrained weights from "{weight_path}"'
        )
    else:
        new_state_dict = OrderedDict()
        matched_layers, discarded_layers = [], []

        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]  # discard module.

            if k in model_dict and model_dict[k].size() == v.size():
                new_state_dict[k] = v
                matched_layers.append(k)
            else:
                discarded_layers.append(k)

        model_dict.update(new_state_dict)
        model.load_state_dict(model_dict)

        if len(matched_layers) == 0:
            LOGGER.warning(
                f'The pretrained weights "{weight_path}" cannot be loaded, '
                "please check the key names manually "
                "(** ignored and continue **)"
            )
        else:
            LOGGER.success(
                f'Successfully loaded pretrained weights from "{weight_path}"'
            )
            if len(discarded_layers) > 0:
                LOGGER.warning(
                    "The following layers are discarded "
                    f"due to unmatched keys or layer size: {*discarded_layers,}"
                )
