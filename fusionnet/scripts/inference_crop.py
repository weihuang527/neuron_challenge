import numpy as np 

class Crop_image(object):
    def __init__(self, img, crop_size=512, overlap=256):
        super(Crop_image, self).__init__()
        self.img = img
        self.crop_size = crop_size
        self.overlap = overlap
        self.dim = len(self.img.shape)
        if self.dim == 3:
            _, self.h, self.w = self.img.shape
        else:
            self.h, self.w = img.shape
        self.pred = np.zeros((self.h, self.w), dtype=np.float32)
        self.add_map = np.zeros((self.h, self.w), dtype=np.float32)
        self.num = (self.h - self.crop_size) // self.overlap + 1
        self.add_crop = np.ones((crop_size, crop_size), dtype=np.float32)
    
    def gen(self, i, j):
        if self.dim == 3:
            raw_crop = self.img[:, i*self.overlap:i*self.overlap+self.crop_size, j*self.overlap:j*self.overlap+self.crop_size]
        else:
            raw_crop = self.img[i*self.overlap:i*self.overlap+self.crop_size, j*self.overlap:j*self.overlap+self.crop_size]
        return raw_crop
    
    def save(self, i, j, pred_crop):
        self.pred[i*self.overlap:i*self.overlap+self.crop_size, j*self.overlap:j*self.overlap+self.crop_size] += pred_crop
        self.add_map[i*self.overlap:i*self.overlap+self.crop_size, j*self.overlap:j*self.overlap+self.crop_size] += self.add_crop
    
    def result(self):
        return self.pred / self.add_map


if __name__ == "__main__":
    import cv2
    from PIL import Image
    img_ = np.asarray(cv2.imread('../data/U-RISC OPEN DATA COMPLEX/train/0189_1_1565791505_73.png', cv2.IMREAD_GRAYSCALE))
    img = np.zeros((10240,10240), dtype=np.uint8)
    img[141:141+9959, 141:141+9958] = img_
    img = img.astype(np.float32) / 255.0

    crop_img = Crop_image(img)
    for i in range(crop_img.num):
        for j in range(crop_img.num):
            raw_crop = crop_img.gen(i, j)
            #########
            pred = raw_crop
            #########
            crop_img.save(i, j, pred)
    results = crop_img.result()

    results = (results * 255).astype(np.uint8)
    results = results[141:141+9959, 141:141+9958]
    Image.fromarray(results).show()
