from pathlib import Path

from PIL import Image

from common import data_path, root_path
from predict import load_model, predict_with_heatmap_for


def main():
    load_model()

    for img_name in LAPTOP_NAMES:
        img_path = data_path() / 'laptop' / img_name
        prediction, heatmap = predict_with_heatmap_for(img_path)
        print(prediction)
        save(heatmap, img_name)

    for img_name in PC_NAMES:
        img_path = data_path() / 'pc' / img_name
        prediction, heatmap = predict_with_heatmap_for(img_path)
        print(prediction)
        save(heatmap, img_name)


def save(heatmap: Image, img_name: str):
    filename = Path(img_name).stem + '-localized' + Path(img_name).suffix
    filepath = root_path() / 'laptoppc' / 'static' / 'output' / filename
    heatmap.save(filepath)


LAPTOP_NAMES = [
    '49def37ba21a4448aee1e46ed8885251.jpg',
    '59d5351caf0f420d959690dfdff63f80.jpg',
    'bcc1f8c6f0884717bee38443b8f966fa.jpg',
    '727c54cd2282484caebcb92863b300e5.jpg',
    '804e0752814748cdb7ddf200e049909f.jpg',
    '21a9ee42fff94b6b99185824ec3c70a5.jpg',
    '4ca6812db71043009bbd62d28d61ed34.jpg',
    '0b5deb6cf6ad46a39fedc540cc4168d9.jpg',
    'f04d30c781154a4794b649890939cae1.jpg',
    'f5353761ee044a0d8c4238222b972c2e.jpg',
    'c99a8b4bbe1b44559790a696364bcd3c.jpg',
    'b80f3282e5d8424b940541fde715437c.jpg',
    'c9d86ae93f89464ea471e3af8c4fd8c7.jpg',
    'c8c3a2ab2f5b45d98af2cc34d9418d06.jpg',
]

PC_NAMES = [
    '5a8e0e6394f3422dba39ba652ee81fbd.jpg',
    '5ac045988d134cd6bbf4b9c690534710.jpg',
    '5f9bca84ff3d44939f8fdde373a97216.jpg',
    '7eaecc5c6b454255a9540e384571887f.jpg',
    '8b118a624072410380ae62fec9286690.jpg',
    '8d83984513bc4bafba93f9d160c2435b.jpg',
    '08ee1aa72b96483894949430d7b21b9d.jpg',
    '9b94bd7df7ea4bb1957d2e2d15db29c4.jpg',
    '9f42b1b8edb04ac38b5628909f83a97e.jpg',
    '20ebd96e7afe47a19e04222c6b59389a.jpg',
]


if __name__ == "__main__":
    main()
