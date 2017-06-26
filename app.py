import os
import time
import cPickle
import datetime
import logging
import flask
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import pandas as pd
from PIL import Image
import cStringIO as StringIO
import urllib
import exifutil

import caffe
import cv2, cv

REPO_DIRNAME = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../..')
UPLOAD_FOLDER = '/tmp/caffe_demos_uploads'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])


app = flask.Flask(__name__)


@app.route('/')
def index():
    return flask.render_template('index.html', has_result=False)


@app.route('/classify_url', methods=['GET'])
def classify_url():
    imageurl = flask.request.args.get('imageurl', '')
    try:
        string_buffer = StringIO.StringIO(
            urllib.urlopen(imageurl).read())
        image = caffe.io.load_image(string_buffer)

    except Exception as err:
        
        
        logging.info('URL Image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open image from URL.')
        )

    logging.info('Image: %s', imageurl)
    result = app.clf.classify_image(image)
    return flask.render_template(
        'index.html', has_result=True, result=result, imagesrc=imageurl)


@app.route('/classify_upload', methods=['POST'])
def classify_upload():
    try:
        
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        imagefile.save(filename)
        im=cv2.imread(filename)
        new=cv2.resize(im, (100, 46))
        

        
        try:
            wavfile = flask.request.files['wavefile']
            wavfilename_ = str(datetime.datetime.now()).replace(' ', '_') + \
                werkzeug.secure_filename(wavfile.filename)
            wavfilename = os.path.join(UPLOAD_FOLDER, wavfilename_)
            wavfile.save(wavfilename)
            imw=cv2.imread(wavfilename)
            neww= cv2.resize(imw, (90, 46))
            
            print("here")
            
            
            
        except Exception as err1:
            print("wave error: %s", err1)
            print(wavfilename)
        
        
        
        
        
        
        
        vis = np.concatenate((new, neww), axis=1)
        cv2.imwrite(filename, vis)
        
        
        
        

        logging.info('Saving to %s.', filename)
        image = exifutil.open_oriented_im(filename)
        
        

    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )

    result = app.clf.classify_image(image)
    return flask.render_template(
        'index.html', has_result=True, result=result,
        imagesrc=embed_image_html(image)
        
    )


def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    image_pil = Image.fromarray((255 * image).astype('uint8'))
    image_pil = image_pil.resize((46, 190))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data


def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
    )


class ImagenetClassifier(object):

    default_args = {
        'model_def_file': (
            '{}/models/mmproject/deploy.prototxt'.format(REPO_DIRNAME)),
        'pretrained_model_file': (

            '{}/snapshot/_iter_100000.caffemodel'.format(REPO_DIRNAME)),
        'class_labels_file': (
            '{}/models/mmproject/label.txt'.format(REPO_DIRNAME))
    }
    for key, val in default_args.iteritems():
        if not os.path.exists(val):
            raise Exception(
                "File for {} is missing. Should be at: {}".format(key, val))
    default_args['image_dim'] = 256
    default_args['raw_scale'] = 255

    def __init__(self, model_def_file, pretrained_model_file, 
                 class_labels_file, gpu_mode,image_dim,raw_scale, mean_file=None, bet_file=None):

        logging.info('Loading net and associated files...')
        if gpu_mode:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        self.net = caffe.Classifier(
            model_def_file, pretrained_model_file, 
            image_dims=(46, 190), mean=np.full((3,46, 190), 127), 
            raw_scale=raw_scale, channel_swap=(2,1,0) 
        )
        
        with open(class_labels_file) as f:
            labels_df = pd.DataFrame([
                {
                    'synset_id': l.strip().split(' ')[0],
                    'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
                }
                for l in f.readlines()
            ])
        
        self.labels = labels_df.sort('synset_id')['name'].values

        
        
        
        
        

    def classify_image(self, image):
        try:
            starttime = time.time()
            
            try:
                scores = self.net.predict([image], oversample=True).flatten()
            except Exception as err1:
                print("ho:"+str(err1))

            endtime = time.time()

            indices = (-scores).argsort()[:5]
            predictions = self.labels[indices]

            
            
            meta = [
                (p, '%.5f' % scores[i])
                for i, p in zip(indices, predictions)
            ]
            


            logging.info('result: %s', str(meta))
            
            

            
            return (True, meta_re, '%.3f' % (endtime - starttime))

        except Exception as err:
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one?')


def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)
    parser.add_option(
        '-g', '--gpu',
        help="use gpu mode",
        action='store_true', default=False)

    opts, args = parser.parse_args()
    ImagenetClassifier.default_args.update({'gpu_mode': opts.gpu})

    
    app.clf = ImagenetClassifier(**ImagenetClassifier.default_args)
    app.clf.net.forward()

    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)

def mergeImage(inputList, output):
    sizeColumn = 2

    image = Image.open(inputList[0])
    x = image.size[0];
    y = image.size[0];
    length = len(inputList);

    newimg=Image.new("RGBA",( int(sizeColumn*(x)) , int((((length-1)/sizeColumn)+1)*(y)) ) )
    print "x,y:",x,y,int(sizeColumn),int(((length-1)/sizeColumn)+1),( int(sizeColumn*(x)) , int((((length-1)/sizeColumn)+1)*(y)) )

    i = 0;
    for j in inputList:
        image = Image.open(j)
        box = (0,0,x,y)
        cutting = image.crop(box)
        print "process:",j,(i/sizeColumn),(i%sizeColumn),box,((x)*(i/sizeColumn),(y)*(i%sizeColumn))
        newimg.paste(cutting,((x)*(i%sizeColumn),(y)*(i/sizeColumn)))
        i=i+1;

    newimg.save(output,"PNG")


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    start_from_terminal(app)
