#Import necessary libraries
from flask import Flask, render_template, request
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import os, sys
from PIL import Image
import numpy as np
from numpy.random import randn, randint
from numpy import asarray
import matplotlib.pyplot as plt
import matplotlib
import uuid

# Create flask instance
app = Flask(__name__)

#Set Max size of file as 10MB.
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

#Allow files with extension png, jpg and jpeg
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init():
    global graph
    graph = tf.compat.v1.get_default_graph()

@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route("/acceuil", methods=['GET', 'POST'])
def acceuil():
    return render_template('home.html')

@app.route("/classification", methods=['GET', 'POST'])
def classification():
    return render_template('classification.html')

@app.route("/predict", methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        try:
            if file and allowed_file(file.filename):
                filename = file.filename
                file_path = os.path.join('static/images/img_pre-prediction', filename)
                file.save(file_path)
                img = load_img(file_path)
                
                if img.mode != "L":
                    image=img.convert("L")
                    image.save('static/images/img_bw/bw_'+ filename )

                image=image.resize((28,28))
                image = img_to_array(image)
                image = image.reshape(1, 28, 28, 1)
                image = image.astype('float32')
                image = image / 255.0
                
                # Predict the class of an image

                with graph.as_default():
                    model1 = load_model('models/classification_model.h5')
                    class_prediction = model1.predict_classes(image)
                    print(class_prediction)

                #Map apparel category with the numerical class
                if class_prediction[0] == 0:
                  product = "T-shirt/top"
                  file_path2 = os.path.join('static/images/Tshirt_top', filename)
                  img.save(file_path2)
                elif class_prediction[0] == 1:
                  product = "Pantalon"
                  file_path2 = os.path.join('static/images/Pantalon', filename)
                  img.save(file_path2)
                elif class_prediction[0] == 2:
                  product = "Pull"
                  file_path2 = os.path.join('static/images/Pull', filename)
                  img.save(file_path2)
                elif class_prediction[0] == 3:
                  product = "Robe"
                  file_path2 = os.path.join('static/images/Robe', filename)
                  img.save(file_path2)
                elif class_prediction[0] == 4:
                  product = "Manteau"
                  file_path2 = os.path.join('static/images/Manteau', filename)
                  img.save(file_path2)
                elif class_prediction[0] == 5:
                  product = "Sandale"
                  file_path2 = os.path.join('static/images/Sandale', filename)
                  img.save(file_path2)
                elif class_prediction[0] == 6:
                  product = "Chemise"
                  file_path2 = os.path.join('static/images/Chemise', filename)
                  img.save(file_path2)
                elif class_prediction[0] == 7:
                  product = "Baskets"
                  file_path2 = os.path.join('static/images/Baskets', filename)
                  img.save(file_path2)
                elif class_prediction[0] == 8:
                  product = "Sac"
                  file_path2 = os.path.join('static/images/Sac', filename)
                  img.save(file_path2)
                else:
                  product = "Bottines"
                  file_path2 = os.path.join('static/images/Bottines', filename)
                  img.save(file_path2)
                return render_template('predict.html', product = product, user_image = file_path2)
        except Exception as e:
            return "Unable to read the file. Please check if the file extension is correct."

    return render_template('predict.html')

@app.route("/generation", methods=['GET', 'POST'])
def generation():
    return render_template('generation.html')

def generate_latent_points(latent_dim, n_samples, n_classes=10):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = randint(0, n_classes, n_samples)
	return [z_input, labels]

def save_plot(examples, n):
    tab_path=[]
    # plot images
    for i in range(n * n):
        # define subplot
        plt.subplot(n, n, 1 + i)
        # turn off axis
        plt.axis('off')
        # save raw pixel data
        path_image=str(uuid.uuid4())+".png"
        plt.imsave("static/images/img_generer/100_sample/"+path_image, examples[i, :, :, 0], cmap='gray_r')
        tab_path.append(path_image) 
    return tab_path
    
def generate_latent_points_one_image(latent_dim, n_samples, lab, n_classes=10):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
    # generate label
	labels=np.array([lab])
	return [z_input, labels]

def save_plot_one_image(examples, n):
    # plot images
    for i in range(n):
        # define subplot
        plt.subplot(n, n, 1 + i)
        # turn off axis
        plt.axis('off')
        #save raw pixel data
        path_image=str(uuid.uuid4())+".png"
        plt.imsave("static/images/img_generer/one_sample/"+path_image, examples[i, :, :, 0], cmap='gray_r')
    return path_image

@app.route("/sample_result", methods=['GET', 'POST'])
def sample_result():
    if request.method == 'POST':

        # generate images
        latent_points, labels = generate_latent_points(100, 100)
        # specify labels
        labels = asarray([x for _ in range(10) for x in range(10)])

        with graph.as_default():
            generator=load_model('models/generator.h5')
            X  = generator.predict([latent_points, labels])
        
        # scale from [-1,1] to [0,1]
        X = (X + 1) / 2.0
        # save the result
        tab_path_img=save_plot(X, 10)
    return render_template('sample_result.html', tab_100_path=tab_path_img)

@app.route("/resultat", methods=['GET', 'POST'])
def resultat():
    if request.method == 'POST':
        categorie = request.form['categorie']

        with graph.as_default():
            generator=load_model('models/generator.h5')

        if(categorie == '0'):
            # generate images
            latent_points, labels = generate_latent_points_one_image(100, 1, 0)
            with graph.as_default():
                # specify labels
                X  = generator.predict([latent_points, labels])
            
            X = (X + 1) / 2.0
            # save the result
            path_img=save_plot_one_image(X, 1)
        elif(categorie == '1'):
            latent_points, labels = generate_latent_points_one_image(100, 1, 1)
            with graph.as_default():
                X  = generator.predict([latent_points, labels])
            X = (X + 1) / 2.0
            path_img=save_plot_one_image(X, 1)
        elif(categorie == '2'):
            latent_points, labels = generate_latent_points_one_image(100, 1, 2)
            with graph.as_default():
                X  = generator.predict([latent_points, labels])
            X = (X + 1) / 2.0
            path_img=save_plot_one_image(X, 1)
        elif(categorie == '3'):
            latent_points, labels = generate_latent_points_one_image(100, 1, 3)
            with graph.as_default():
                X  = generator.predict([latent_points, labels])
            X = (X + 1) / 2.0
            path_img=save_plot_one_image(X, 1)
        elif(categorie == '4'):
            latent_points, labels = generate_latent_points_one_image(100, 1, 4)
            with graph.as_default():
                X  = generator.predict([latent_points, labels])
            X = (X + 1) / 2.0
            path_img=save_plot_one_image(X, 1)
        elif(categorie == '5'):
            latent_points, labels = generate_latent_points_one_image(100, 1, 5)
            with graph.as_default():
                X  = generator.predict([latent_points, labels])
            X = (X + 1) / 2.0
            path_img=save_plot_one_image(X, 1)
        elif(categorie == '6'):
            latent_points, labels = generate_latent_points_one_image(100, 1, 6)
            with graph.as_default():
                X  = generator.predict([latent_points, labels])
            X = (X + 1) / 2.0
            path_img=save_plot_one_image(X, 1)
        elif(categorie == '7'):
            latent_points, labels = generate_latent_points_one_image(100, 1, 7)
            with graph.as_default():
                X  = generator.predict([latent_points, labels])
            X = (X + 1) / 2.0
            path_img=save_plot_one_image(X, 1)
        elif(categorie == '8'):
            latent_points, labels = generate_latent_points_one_image(100, 1, 8)
            with graph.as_default():
                X  = generator.predict([latent_points, labels])
            X = (X + 1) / 2.0
            path_img=save_plot_one_image(X, 1)
        elif(categorie == '9'):
            latent_points, labels = generate_latent_points_one_image(100, 1, 9)
            with graph.as_default():
                X  = generator.predict([latent_points, labels])
            X = (X + 1) / 2.0
            path_img=save_plot_one_image(X, 1)

    return render_template('resultat.html',Label=categorie, path=path_img) 

@app.route("/tshirt_top", methods=['GET', 'POST'])
def tshirt_top():
    image_names = os.listdir('static/images/Tshirt_top/')
    nb=len(image_names)
    return render_template('tshirt_top.html', image=image_names, nb_products=nb)

@app.route("/pantalon", methods=['GET', 'POST'])
def pantalon():
    image_names = os.listdir('static/images/Pantalon/')
    nb=len(image_names)
    return render_template('pantalon.html', image=image_names, nb_products=nb) 

@app.route("/pull", methods=['GET', 'POST'])
def pull():
    image_names = os.listdir('static/images/Pull/')
    nb=len(image_names)
    return render_template('pull.html', image=image_names, nb_products=nb)  

@app.route("/robe", methods=['GET', 'POST'])
def robe():
    image_names = os.listdir('static/images/Robe/')
    nb=len(image_names)
    return render_template('robe.html', image=image_names, nb_products=nb)       

@app.route("/manteau", methods=['GET', 'POST'])
def manteau():
    image_names = os.listdir('static/images/Manteau/')
    nb=len(image_names)
    return render_template('manteau.html', image=image_names, nb_products=nb)    

@app.route("/sandale", methods=['GET', 'POST'])
def sandale():
    image_names = os.listdir('static/images/Sandale/')
    nb=len(image_names)
    return render_template('sandale.html', image=image_names, nb_products=nb)

@app.route("/chemise", methods=['GET', 'POST'])
def chemise():
    image_names = os.listdir('static/images/Chemise/')
    nb=len(image_names)
    return render_template('chemise.html', image=image_names, nb_products=nb)

@app.route("/baskets", methods=['GET', 'POST'])
def baskets():
    image_names = os.listdir('static/images/Baskets/')
    nb=len(image_names)
    return render_template('baskets.html', image=image_names, nb_products=nb)

@app.route("/sac", methods=['GET', 'POST'])
def sac():
    image_names = os.listdir('static/images/Sac/')
    nb=len(image_names)
    return render_template('sac.html', image=image_names, nb_products=nb)

@app.route("/bottines", methods=['GET', 'POST'])
def bottines():
    image_names = os.listdir('static/images/Bottine/')
    nb=len(image_names)
    return render_template('bottine.html', image=image_names, nb_products=nb)

@app.route("/aboutus", methods=['GET', 'POST'])
def aboutus():
    return render_template('aboutus.html')

if __name__ == "__main__":
    init()
    app.debug=True
    app.run()