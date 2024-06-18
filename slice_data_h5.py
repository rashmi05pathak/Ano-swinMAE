import os
import glob
import numpy as np
import torch
from tqdm.notebook import tqdm
from monai.data import ITKReader,NibabelReader
from monai.transforms import LoadImage, LoadImaged
from monai.transforms import (
    Orientationd, EnsureChannelFirst, Compose, ToTensord, Spacingd,Resized,ScaleIntensityD,ResizeWithPadOrCropd
    # ScaleIntensityD, ScaleIntensityRangeD, AdjustContrastD, RandAffined, ToNumpyd,RepeatChannelD
)
from monai.data import Dataset
import h5py
import threading

class H5CachedDataset(Dataset):
    def __init__(self,niilist,masklist,
                 transforms_offline, transforms_online=ToTensord(('image',)),
                 nslices_per_image = 362 ,
                 start_slice = 61,
                 end_slice = 91,
                 h5cachedir=None):
        #### nslices_per_image ---> total slice in the volume
        #### h5cachedir ---> directory to save one .h5 files for each volume & it would act like a cache directory
        #### if h5cachedir does not exist create one 

#         self.lock = threading.Lock()
        if h5cachedir is not None:
            if not os.path.exists(h5cachedir):
                os.mkdir(h5cachedir)
            self.cachedir = h5cachedir
        #### datalist ---> a list [{'image': volume_1_path},......,{'image': volume_n_path}]
        #### masklist ---> a list [{'mask': mask_1_path},......,{'image': mask_n_path}]
        self.datalist = niilist
        self.masklist = masklist
        self.xfms = transforms_offline
        self.xfms2 = transforms_online
        #### 3d image loader from monai
        self.loader = LoadImage()
        self.loader.register(NibabelReader())  
        #### start_slice & end_slice---> slices to be truncated in each volume vol[:,:,start_slice:-end_slice]
        self.start_slice = start_slice
        self.end_slice = end_slice
        #### nslices ---> nslices_per_image - end_slice i.e. slice value after end truncation
        #            ---> nslices is kept flexible so that index is obtained by adding front truncation value &
        #            ---> total length of the loder is caluclated considering subtracting front truncation value
        self.nslices = nslices_per_image - self.end_slice
        
    def __len__(self):
        #### total number of slices in all the volumes
        return len(self.datalist)*(self.nslices - self.start_slice)
    
    def clear_cache(self):
        #### function to clear the directory storing h5 files (used for caching the h5 files)
        for fn in os.listdir(self.cachedir):
            os.remove(self.cachedir+'/'+fn)
            
    def __getitem__(self,index):
        #### ditionary to store data slicewise
        data = {}
        #### index can take values from 
                # 0 to (total number of volumes * (len(datalist)*(nslices - start_slice)))  
        #### filenum can take values from 0 to total number of volumes
        #### slicenum can take values from 0 to (len(datalist)*(nslices - start_slice))
        filenum = index // (self.nslices - self.start_slice)
        slicenum = index % (self.nslices - self.start_slice)
        slicenum += self.start_slice
        #### Extract the datafile location & mask file location based on filenum
        datalist_filenum = self.datalist[filenum]
        loc_data = datalist_filenum['image']
        masklist_filenum = self.masklist[filenum]
        loc_mask = masklist_filenum['label']
        
        


        ##### if h5 exists for the current volume fill data dictionary with current slice number
        if self.cachedir is not None:
            h5name = self.cachedir+'/%d.h5' % filenum
            ptname = self.cachedir+'/%d.pt' % filenum

            if os.path.exists(h5name):
                
                with h5py.File(h5name,'r',libver='latest', swmr=True) as itm:
                    for key in itm.keys():                       
                        data[key]=torch.from_numpy(itm[key][:,:,:,slicenum])
                data['image_meta_dict']={'affine':np.eye(3)} # FIXME: loses spacing info - can read from pt file


        ##### if data dictionary is empty
        if len(data)==0:
           
            #### Read image & mask data, meta data
            print(loc_data)
            imgdata = self.loader(loc_data) 
            meta = imgdata
            mask_data = self.loader(loc_mask)
            mask_meta = mask_data
            #### store volume wise image & mask data,metadata in a dictionary 
            data_i = {'image':imgdata,'label':mask_data, 'image_meta_dict':meta, 'label_meta_dict':mask_meta}
            #### transform the data dictionary
            data3d = self.xfms(data_i)
            #### Create h5 file for the volume by chunking into the slice shape for data & mask 
            #### Create a .pt file for meta data
            if self.cachedir is not None:
                other = {}

                with h5py.File(h5name,'w',libver='latest') as itm:
                    itm.swmr_mode = True
                    for key in data3d:
                        if key=='image' or key=='label':                             
                            img_npy = data3d[key].numpy()
                            if key == 'label':
                                img_npy = (img_npy>0).astype('uint8')
                            shp = img_npy.shape
                            chunk_size = list(shp[:-1])+[1]
                            ds = itm.create_dataset(key,shp,chunks=tuple(chunk_size),dtype=img_npy.dtype)
                            ds[:]=img_npy[:]
                            ds.flush()
                    else:
                        other[key]=data3d[key]
                torch.save(other,ptname)


            #### fill the data dictionary
            data = {
                'image':data3d['image'][:,:,:,slicenum],
                'label':data3d['label'][:,:,:,slicenum],
                'image_meta_dict':{
                    'affine':np.eye(3)
                }
            }

            
        if len(data)>0:
#             print("**",data.keys())
#             res = self.xfms2(data)
            res = data
            res['image']=res['image'].float()
            res['label']=res['label'].to(torch.int64)
            res['filenum'] = filenum
            res['filepath'] = loc_data
            res['slicenum'] = slicenum
            res['idx'] = index
            return res

        else:
            # replace with random
            return self.__getitem__(np.random.randint(len(self.datalist)))
