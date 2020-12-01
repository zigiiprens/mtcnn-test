# MTCNN-TEST
## Description
This repository aims to test the MTCNN package from Pypi https://pypi.org/project/mtcnn/

## Installation
We, for now, advice to clone this repository in your Projects folder and then run
* `pip install -r requirements.txt`

Make sure you have installed all the required packages or this will not work properly.
If needed, you can use an virtual env manager as:
* `venv`
* `pipenv`

## Folders
In the `mtcnn-test` project repository, you will need an extra folder inside as:
* `data`
* `data/input`
* `data/output`

Make sure to move an image in the `data/input` folder and rename it as `face.jpg`

## Run the magic
Finally, you will just need to run the following code with a simple line as:
* `python FaceDetection.py --image <path>`

If you do not precise a `--image <path>`, the script will automatically use `data/input/face.jpg`.

Now, see the Magic.

## Authors
Samir Omar (zigiiprens)

## LICENSE
MIT License

Copyright (c) 2020 Samir OMAR

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.