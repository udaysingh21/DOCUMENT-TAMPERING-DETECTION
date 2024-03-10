import pandas as pd
import gradio as gr
from ela_data_generator import generate_ela_batch, generate_data, preprocessing
from keras.models import load_model

'''
Original : 0
Tampered : 1
'''


# Create a function to predict the image
def predicting_fun(input_image):
    # Genrating ela image
    ela_image=generate_ela_batch(input_image)

    # Adjusting the size and channels of both images and converting them to numpy arrays
    input_image,ela_image=preprocessing(input_image,ela_image)

    # Generating the data for input image
    data=generate_data(input_image,ela_image)
    print(data)

    # Loading the model
    model=load_model('model.h5')

    xin=pd.DataFrame(data)
    img_class=model.predict(xin)

    # print(img_class)
    img_class=(img_class>0.5).astype("int32")

    print(img_class[0][0])
    # return ela_image

    if img_class[0][0]==0:
        return "Original"
    else:
        return "Tampered"    


   

# Create a interface
ui=gr.Interface(
    fn=predicting_fun, # function to predict the image
    inputs=gr.Image(type="pil",label="Upload Document"), # input image
    title='Document Tampering Detection',
    outputs='text', 
)

if __name__ == "__main__":
    ui.launch()