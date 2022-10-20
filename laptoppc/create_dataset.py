import logging as log

import DuckDuckGoImages as ddg

from common import data_path

"""
This could be used to re-create data/ directory from scratch. However, lots of manual effort went into fixing
categories of downloaded images. After re-download you would need to manually verify and correct all downloaded images.
"""


def main():
    setup_logging()
    #create_dataset(['laptop', 'pc'])
    remove_fake_jpegs()


def create_dataset(categories):
    for category in categories:
        cat_path = data_path() / category
        cat_path.mkdir(exist_ok=True, parents=True)
        ddg.download(category, cat_path, max_urls=500, parallel=True)


def remove_fake_jpegs():
    import imghdr
    paths = list(data_path().glob('**/*'))
    for path in paths:
        if path.is_file() and imghdr.what(path) != 'jpeg':
            log.info(f'Removing fake JPEG file: {path}')
            path.unlink()


def setup_logging():
    log.basicConfig(level=log.INFO)


if __name__ == '__main__':
    main()
