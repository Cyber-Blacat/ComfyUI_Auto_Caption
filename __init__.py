
from .auto_caption import Joy_Model_load, Auto_Caption
from .auto_caption import LoadManyImages
from .auto_caption2 import ExtraOptionsSet, Auto_Caption2, Joy_Model2_load

NODE_CLASS_MAPPINGS = {
    "Joy Model load":Joy_Model_load,
    "Auto Caption":Auto_Caption,
    "LoadManyImages":LoadManyImages,
    "Joy_Model2_load": Joy_Model2_load,
    "ExtraOptionsSet": ExtraOptionsSet,
    "Auto_Caption2": Auto_Caption2,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Joy Model load":"Joy Model load",
    "Auto Caption":"Auto Caption",
    "LoadManyImages":"Load Many Images",
    "Joy_Model2_load":"Joy caption 2 model loader",
    "ExtraOptionsSet":"Extra Options Set",
    "Auto_Caption2":"Auto Caption 2",
}