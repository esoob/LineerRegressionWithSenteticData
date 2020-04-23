import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import pathlib
from matplotlib import pyplot as plt




def build_model(my_learning_rate):
    """Öncelikle modeli oluşturan ve eğiten modelleri tanımlıyoruz."""

     #basit tf.keras model tipi çoğunlukla sequential(sıralıdır.)
     #Sıralı bir model bir vya daha çok label (katman) içerir.
    model = tf.keras.models.Sequential()


    #Modelin topografisini tanımla.
    #basit bir lineer regression modelinin topografisi
    #tek bir katmandaki(layer) tek bir düğümdür(node)
    #dense layer girdideki her bir düğüm çıkıştaki
    #her bir düğüm ile bağlıdır.
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

    #model topografisini TensorFlow'un verimli bir şekilde
    #yürütebileceği(execute) koda dönüştür. Modelin
    #mean squared error minimize etmek(en aza indirmek) için
    #eğitimi yapılandır. Optimizer=ağırlık katsayılarının
    #güncellenmesi için kullanılacak optimizasyon yöntemi.
    #loss function: gerçek değer ile tahmin edilen değer
    #arasındaki hatayı ifade eden metrik. metrics: eğitim
    #ve test sırasındaki değerlendirme parametreleri.
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),
                   loss = "mean squarred_error",
                   metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

def train_model(model, feature, label, epoch, batch_size):
     history = model.fit(x=feature,
                         y=label,
                         batch_size=None,
                         epoch=epochs)

     #eğitilmiş modelin weight(ağırlık) ve bias(sapma) al.
     trained_weight = model.get_weights()[0]
     trained_bias = model.get_weights()[1]
     
     epochs = history.epoch
     hist = pd.DataFrame(history.history)
     rmse = hist["root_mean_squared_error"]
     return trained_weight, trained_bias, epochs, rmse

print("Defined create_model and train_model")

def plot_the_model(trained_weight, trained_bias, feature, label):
    """Eğitilmiş modelin eğitilen feature ve label bağlı grafiğini çizer"""

    #Eksenleri tanımla.
    plt.xlabel("feature")
    plt.ylabel("label")

    #feature değerlerini ve label değerlerinin grafiğini çiz.
    plt.scatter(feature, label)

    #Modeli temsil eden kırmızı çizgi oluştur. Kırmızı çizgi
    #(x0, yo) koordinatlarından başlar ve (x1, y1) koordinatlarında biter.
    x0 = 0
    y0 = trained_biasx1=my_feature[-1]
    y1 = trained_bias + (trained_weight * x1)
    #plot() çok yönlü bir fonksiyondur ve isteğe bağlı sayıda argüman alır.
    plt.plot([x0,x1], [y0,y1], c='r')
    #graafiği ve kırmızı çizgiyi resimle
    plt.show()

def plot_the_loss_curve(epochs, rmse):
    """epoch'a karşı kaybı gösteren kayıp eğrisini çizmek"""

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs, rmse, label="Loss")
    plt.legend()
    plot.ylim([rmse.min()*0.97, rmse.max()])
    plt.show()

print("Defined the plot_the_model and plot_the_loss_curve functions")

#dataset
my_feature = ([1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0, 10.0, 11.0, 12.0])
my_label = ([5.0, 8.8,  9.6, 14.2, 18.8, 19.5, 21.4, 26.8, 28.9, 32.0, 33.8, 38.2])

#hyperparamereleri başlat ve modeli oluşturan ve eğiten işlevleri çağır.
learning_rate = 0.01
epoch=10
my_batch_size=12

my_model = build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(my_model, my_feature,
                                                         my_label, epochs,
                                                         my_batch_size)
plot_the_model(trained_weight, trained_bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)




