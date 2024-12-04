import pydicom
import glob
import numpy as np
import re
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


class ConvertModel:
    def __init__(self, patient_id):
        self.patient_id = patient_id
    
    def construct_model(self):
        from matplotlib import pyplot as plt
        from sklearn.neighbors import KNeighborsClassifier
        #knnで拡張
        n=1
        rectype = np.float64

        knn_x=KNeighborsClassifier(n_neighbors=n)
        knn_y=KNeighborsClassifier(n_neighbors=n)
        knn_z=KNeighborsClassifier(n_neighbors=n)

        plot=0
        
        lis = glob.glob(f'/mnt/e/dataset/LUNG1-{self.patient_id}/original_data/*.dcm')
        first_slice = lis[0]

        with open(f'/mnt/e/dataset/LUNG1-{self.patient_id}/LUNG1-{self.patient_id}.bin', 'rb')as f:
            data = np.fromfile(f, dtype=np.uint8).reshape(512, 512, len(lis))

        # pydicomを使用してPixelSpacingとSliceThicknessを取得
        first_slice = pydicom.dcmread(first_slice)

        # parmeter settings
        Nx=600
        Ny=600
        Nz=600
        Nxt=512
        Nyt=512
        Nzt=len(lis)
        PixelSpace=first_slice.PixelSpacing[0]
        Thickness=first_slice.SliceThickness

        dx=0.5
        dy=0.5
        dz=0.5


        x=np.array(np.arange(Nx)*dx,dtype=rectype).reshape(-1,1)
        xt=np.array(np.arange(Nxt)*PixelSpace,dtype=rectype).reshape(-1,1)
        y=np.array(np.arange(Ny)*dy,dtype=rectype).reshape(-1,1)
        yt=np.array(np.arange(Nyt)*PixelSpace,dtype=rectype).reshape(-1,1)
        z=np.array(np.arange(Nz)*dz,dtype=rectype).reshape(-1,1)
        zt=np.array(np.arange(Nzt)*Thickness,dtype=rectype).reshape(-1,1)
        nums=np.array(np.arange(Nxt),dtype=rectype).reshape(-1,1)
        nums_z=np.array(np.arange(Nzt),dtype=rectype).reshape(-1,1)

        #拡張
        knn_x.fit(xt,np.ravel(nums))
        IDXx=knn_x.predict(x).astype(int)
        knn_y.fit(yt,np.ravel(nums))
        IDXy=knn_y.predict(y).astype(int)
        knn_z.fit(zt,np.ravel(nums_z))
        IDXz=knn_z.predict(z).astype(int)
        

        tempz1=np.zeros((Nxt,Nyt,Nz),dtype=rectype)

        for i in range(Nz):
            tempz1[:,:,i]=data[:,:,int(IDXz[i])]

        tempx1=np.zeros((Nx,Nyt,Nz),dtype=rectype)

        for i in range(Nx):
            tempx1[i,:,:]=tempz1[int(IDXx[i]),:,:]
            
        tempy1=np.zeros((Nx,Ny,Nz),dtype=rectype)
        
        for i in range(Ny):
            tempy1[:,i,:]=tempx1[:,int(IDXy[i]),:]

        model1 = tempy1.astype(np.uint8)
    

        print(model1.shape)
        if plot:
            plt.imshow(model1[:,255,:])
            plt.show()

            plt.imshow(model1[:,:,300])
            plt.show()

            plt.imshow(model1[400,:,:])
            plt.show()
            for i in range(10):
                plt.imshow(model1[:,255+i*10,:])
                plt.show()
                
        self.model1 = model1
        model1.tofile(f"/mnt/e/dataset/LUNG1-{self.patient_id}/{self.patient_id}_tumour_model.bin")
    
    def plot_model(self):
        from mayavi import mlab
        mlab.figure()
        mlab.contour3d(self.model1, contours=10, opacity=0.1)
        mlab.show()
        

if __name__ == '__main__':
    patient_id = '089'
    res = ConvertModel(patient_id)
    res.construct_model()
    # res.plot_model()
