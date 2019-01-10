from model import *
from data import *
 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
 
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

myGene = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)
#得到一個訓練生成器，以batch=2的速率無限生成增強後的數據
 
model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
#回調函數，第一個是保存模型路徑，第二個是檢測的值，檢測Loss是使它最小，第三個是只保存在驗證集上性能最好的模型
 
model.fit_generator(myGene,steps_per_epoch=1000,epochs=1000,callbacks=[model_checkpoint])
#steps_per_epoch指的是每個epoch有多少個batch_size，也就是訓練集總樣本數除以batch_size的值
#上面一行是利用生成器進行batch_size數量的訓練，樣本和樣本標籤myGene傳入

#model = unet('unet_membrane.hdf5')


testGene = testGenerator("data/membrane/test")
results = model.predict_generator(testGene,64,verbose=1)
saveResult("data/membrane/test",results)
#30是step,steps: 在停止之前，来自 generator 的總步數 (樣本批次)。 可選参数 Sequence：如果未指定，就使用len(generator) 作為步數
#上面的數return值是：預測值的 Numpy 。
  