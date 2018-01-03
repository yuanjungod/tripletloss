import caffe
import os
import numpy as np

caffe.set_mode_gpu()
deploy = os.path.join("../", 'deploy.prototxt')
# caffemodel = os.path.join("../models/", 'vggnet_pretrained.caffemodel')
caffemodel = os.path.join("../models/", '10.caffemodel')

print("load model begin")
net_full_conv = caffe.Net(deploy, caffemodel, caffe.TEST)


# net_full_conv.blobs['data'].reshape(1,3,scale_img.shape[1],scale_img.shape[0])
net_full_conv.blobs['data'].reshape(1, 3, 224, 224)
transformer = caffe.io.Transformer({'data': net_full_conv.blobs['data'].data.shape})

# ilsvrc_2012_mean = os.path.join("../models/", 'ilsvrc_2012_mean.npy')
# transformer.set_mean('data', np.load(ilsvrc_2012_mean).mean(1).mean(1))
# transformer.set_mean('data', np.array([[102.9801, 115.9465, 122.7717]]))
transformer.set_mean('data', np.array([102.9801, 115.9465, 122.7717]))
transformer.set_transpose('data', (2, 0, 1))
transformer.set_channel_swap('data', (2, 1, 0))
# transformer.set_raw_scale('data', 255.0)

for img_path in os.listdir("D:/data/face_data/tripletloss_face/"):
    print(img_path)
    # img_path = "D:/face_data/Aaron_Patterson@0017.jpg"
    im = caffe.io.load_image("D:/data/face_data/tripletloss_face/"+img_path)
    out = net_full_conv.forward_all(data=np.asarray([transformer.preprocess('data', im)]))
    print(out["fc9_1"].shape)

