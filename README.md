# Dermoscopic Analyzer ##

A simple analyzer for dermoscopic images implemented by pytorch. The frontedge is inspired by the project flask-image-upload.

## Introduce ###

The model is trained with ISIC dataset. Now it can classify 15 different kinds of skin disease.

- actinic keratosis (光线性角化病),
- angioma (血管瘤),
- atypical melanocytic proliferation (非典型黑色素细胞增生),
- basal cell carcinoma (基底细胞上皮瘤),
- dermatofibroma (皮肤纤维瘤),
- lentigo NOS (雀斑痣),
- lentigo simplex (单纯性雀斑痣),
- melanoma (黑素瘤), 
- nevus (痣),
- other (其他),
- pigmented benign keratosis (色素性良性角化病),
- seborrheic keratosis (脂溢性角化病),
- solar lentigo (日光性着色班),
- squamous cell carcinoma (鳞状细胞癌),
- vascular lesion (血管病变).

## Usage ###

Install some dependencies. And run the train.py to train the model pretrained by imagenet.
```bash
pip3 install -r requirement.txt
python3 train.py --cuda --root path/to/dir
```

Then run the app.py to test the Flask server.
```bash
python3 app.py
```

You can deploy the app to the production environment using gunicorn.
```bash
export PORT=8000
gunicorn --worker-class=gevent --timeout 300 --bind 0.0.0.0:$PORT app:APP
``` 
