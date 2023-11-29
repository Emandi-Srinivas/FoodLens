# FOOD LENS 🌯🎥

Foodlens is an advanced end-to-end **CNN Image Classification project** that comes with Keras We have created our own **FOOD LENS** Model inspired from the EfficientNetB1 model to accurately identify different types of food items from images.

The model has been trained on the Food101 dataset. It can identify over 100 different food classes.

The project also includes a user-friendly web application with a neat front-end built with **Streamlit**, allowing users to upload food images and receive instant predictions along with the **top-5 predictions** from the model.

**Accuracy :** **`85%`**

**Model :** **`FOOD LENS`**

**Dataset :** **`Food101`**

## Looks Great, How you can use it?

Finally, after training the model, we exported it as `.hdf5` files and then integrated it with **Streamlit Web App**. 

**Streamlit** turns data scripts into shareable web apps in minutes.

Once the web app is loaded,

1. Upload an image of food. If you do not have one, just use the images from `online`, drag and drop is working.
2. Once the image is processed, the **`Predict`** button appears. Click it.
3. Once you click the **`Predict`** button, the model prediction takes place and the output will be displayed along with the model's **Top-5 Predictions**
4. And That's it, there you go.

> If anyone wants to run it in the local host by downloading the whole package from Github. They need these modules 👇

1. Streamlit
2. Tensorflow
3. Requests            # pip install requests
4. streamlit_lottie    # pip install streamlit-lottie

We can download these modules by various like pip etc.

## Okay Cool, How did we build it?

1. #### Imported Food101 dataset from **[Tensorflow Datasets](https://www.tensorflow.org/datasets)** Module.

2. #### Becoming one with the Data : *Visualise - Visualize - Visualize*

3. #### Setup Global type policy to **`mixed_float16`** to implement [**Mixed Precision Training**](https://www.tensorflow.org/guide/mixed_precision)

   > Mixed precision is the use of both 16-bit and 32-bit floating-point types in a model during training to make it **run faster** and use **less memory**.

4. #### Building the Model Callbacks 

   As we are dealing with a complex Neural Network (EfficientNetB0) it's a good practice to have a few callbacks set up. A few ones I will be using throughout this Notebook are :

   - **TensorBoard Callback:** TensorBoard provides the visualization and tooling needed for machine learning experimentation

   - **EarlyStoppingCallback:** Used to stop training when a monitored metric has stopped improving.

   - **ReduceLROnPlateau:** Reduce learning rate when a metric has stopped improving.


5. #### Built a  [Fine Tuning](https://www.tensorflow.org/tutorials/images/transfer_learning)  Model

   This part took the longest. In Deep Learning, you have to know which nob does what. Once you get experienced you'll what nobs you should turn to get the results you want. 
   **Architecture** : **`EffficientNetB1`**

6. #### Evaluating and Deploying out Model to Streamlit

   Once we have our model ready, it's crucial to evaluate it on our **custom data** : *the data our model has never seen*.

   Training and evaluating a model on train and test data is cool, but making predictions on our own real-time images is another level.

   Once we are satisfied with the results, we can export the model as a `.hdf5`  which can be used in the future for model deployment.

Once the model is exported then there comes the Deployment part. Check out  **`app.py`**

We've used **Lottie** in our project, Lottie is a library that parses **Adobe After Effects animations** exported as JSON with Bodymovin(After Effects composition embedding extension) and renders them natively on the web.

## Breaking down the repo

At first glance, the files in the repo may look intimidating and overwhelming. To avoid that, here is a quick guide :

* `.gitignore`: tells what files/folders to ignore when committing.
* `app.py`: Our Food Vision app was built using Streamlit.
* `utils.py`: Some of the used functions in  `app.py`.
* `final_Prepared_Model_with_custom_data.ipynb`: Google Colab Notebook used to train the model.
* `models`: Contains all the models used as *.hdf5* files.

## Created by - 

*  Emandi Srinivas
*  Vishal Saini
*  MD Ashif Jamal
*  Aditya Singh
*  Ishu Malik
*  Yuvraj Choudhary
*  KeeratPreet Singh


######                                             *Inspired by **Daniel Bourke's** CS329s Lecture*
