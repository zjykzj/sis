# -*- coding:utf-8-*-

import logging

from PIL import Image
from datetime import datetime
from flask import Flask, request, render_template
from gevent import pywsgi

from search import Search

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

model = Search()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        scores = model.run(img)

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')


if __name__ == "__main__":
    # app.run("0.0.0.0")
    server = pywsgi.WSGIServer(('0.0.0.0', 12375), app)
    server.serve_forever()
