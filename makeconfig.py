import os
import configparser

root = os.getcwd()

config = configparser.ConfigParser()

config['paths'] = {
    'model_weights': os.path.join("models", "yolo_models"),
    'folder_output_matrices':  "output_folder",
    'summary_matrix' : "summary_matrix.npy",
    'folder_input_imgs' : os.path.join("data", "input_images"),
    'folder_classificators':os.path.join("models", "classificators")
}

config["points"] = {
    "left leg":[12, 14, 16],
    "right leg":[11, 13, 15],
    "left hand":[6, 8, 10],
    "right hand":[5, 7, 9]
}

config["hyperparams"] ={
    'grid_step' : '15', # шаг сетки заполнения пустого пространства
    'points_conf_level' : '0.7', # вклад попадания точек в свои области, тогда вклад совпадения углов = 1 - points_conf_level
    'std_for_angle' : '15', # Градусное отклонение от модели
    'kernel_SVC':"rbf", #Ядро
    'gamma_SVC':'0.01',
    'C_SVC':'1.0'
}

if __name__ == '__main__':

    conf_path = os.path.join(root, "configfiles")
    os.makedirs(conf_path, exist_ok=True)
    with open(os.path.join(conf_path, "conf.ini"), 'w') as configfile:
        config.write(configfile)