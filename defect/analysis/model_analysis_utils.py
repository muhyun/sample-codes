import os

from mxnet import gluon, image, nd
from mxnet.gluon.data.vision import transforms
from gluoncv.model_zoo import get_model

import cv2

import numpy as np

rootdir = '/home/ec2-user/SageMaker'
model_dir = f'{rootdir}/AmazonSageMaker-defect-classifier-poca/models/patch_classification'

models = {
    'exp26': 'ResNet18_v2-GI-datasetv2-26-0827_124359.params',
    'exp27': 'ResNet18_v2-GA-datasetv2-27-0827_135338.params'
    
}
model_names = ['ResNet18_v2']

def get_labels(camera_type='GA', version='2'):
    dataset_dir = f'{rootdir}/dataset_v{version}/datasets-patches-1/{camera_type}'    
    train_path = os.path.join(dataset_dir, 'train')
    train_ds = gluon.data.vision.ImageFolderDataset(train_path)
    
    return train_ds.synsets
    
    
class ModelAnalysisUtil:
    
    camera_types = ['GA', 'GI']
    versions = ['2', '3']
    labels = {}

    def __init__(self):
        # set labels for datasets
        for camera_type in self.camera_types:
            for version in self.versions:
                self.labels[f'{camera_type}_v{version}'] = get_labels(camera_type, version)
                
        # set test data transforms
        
        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def get_model(self, exp_no):
        model_fname = models[exp_no]
        model_full_fname = f'{model_dir}/{model_fname}'
            
        camera_type = 'GA' if 'GA' in model_fname else 'GI'
        version = model_fname.split('-')[2].replace('datasetv','')
        num_classes = len(self.labels[f'{camera_type}_v{version}'])
                    
        model_full_fname = f'{model_dir}/{model_fname}'
        net = self.get_net(model_fname, num_classes)
        
        net.load_parameters(model_full_fname)
        
        return net, camera_type, version
    
    def get_model_name(self, exp_no):
        return f'{exp_no}-{models[exp_no]}'

    def get_net(self, model_fname, classes):
        for model_name in model_names:
            if model_name in model_fname:
                net = get_model(model_name, classes=classes, pretrained=False)
        
        return net
    
    def get_predicted_label(self, net, image_fname, camera_type, version):
        y_true_label = image_fname.split('/')[-2]
        img = image.imread(image_fname)
        img = self.transform_data(img)
        batch = img.expand_dims(axis=0)

        y_pred = net(batch)[0]
        y_pred_index = int(np.argmax(y_pred).asscalar())        
        y_pred_label = self.get_label_name(y_pred_index, camera_type, version)

        return y_true_label, y_pred_label
                
    def get_label_name(self, label_index, camera_type, version):
        return self.labels[f'{camera_type}_v{version}'][label_index]
    
    def transform_data(self, X, camera_types='GA', versions='2'):
        return self.transforms(X)
        

    def get_CAM(self, net, image_fname, camera_type, version):
        
        img = image.imread(image_fname)
        img = self.transforms(img)
        img = img.expand_dims(0)
        feature_conv = net.features[:-4](img).asnumpy()
        
        output = net(img)
        idx = nd.argmax(output).asscalar().astype('int32')
        
        params_list = list(net.collect_params())
        weight_softmax = net.collect_params()[params_list[-2]].data().asnumpy()
        
        size_upsample = (128, 224)
        bz, nc, h, w = feature_conv.shape
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        
        return cv2.resize(cam_img, size_upsample)
    
    def get_CAM_heatmap(self, net, image_fname, camera_type, version):
        CAMs = self.get_CAM(net, image_fname, camera_type, version)
        
        img = cv2.imread(image_fname)
        height, width, _ = img.shape
        heatmap = cv2.applyColorMap(cv2.resize(CAMs,(width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + img * 0.5
        
        cam_heatmap_fname = './cam_heatmap.jpg'
        cv2.imwrite(cam_heatmap_fname, result)
        
        return cam_heatmap_fname