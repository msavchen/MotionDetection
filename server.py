from flask import Flask, request, render_template
import os


app = Flask(__name__)
ON = 0
CLASSES = ["aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "cow", "dog",
        "horse", "motorbike", "person", "sheep", "train"]

@app.route('/', methods=['GET', 'POST'])
def index():
    global ON
    if request.method == 'POST':
        class_list = request.form.getlist("class")
        class_list = prepare_classes(class_list)
        if ON == 1:
            os.system("cd /home/pi/Desktop/pi-object-detection")
            os.system("pkill -f real_time_object_detection.py")
        os.system("cd /home/pi/Desktop/pi-object-detection")
        os.system("python3 real_time_object_detection.py -cl " + class_list)
        ON = 1
    return render_template('config.html', classes = CLASSES)

def prepare_classes(class_list):
    arg = ""
    for c in class_list:
        arg += c + " "
    return arg

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
    