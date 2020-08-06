
import keras

model = keras.models.load_model('iris_model')



from ann_visualizer.visualize import ann_viz;
ann_viz(model, view=True, filename='NN.pdf', title='iris_NN')
