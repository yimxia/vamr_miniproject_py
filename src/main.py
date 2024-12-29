from src.utils import utils
from src.utils.image_reader import KittiDataReader, MalagaDataReader, ParkingDataReader


def demo_image_readers() -> None:
    KittiDataReader.show_image()
    MalagaDataReader.show_images(end_id=4)
    ParkingDataReader.show_image(id=5)

    image1 = KittiDataReader.read_image()
    image2 = MalagaDataReader.read_image(id=3)
    images = ParkingDataReader.read_images(start_id=100, end_id=105)

    for image in [image1, image2, *images]:
        utils.show_img(img=image.img)


def main() -> None:
    pass


if __name__ == "__main__":
    demo_image_readers()
    main()
