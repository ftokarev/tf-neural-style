# Neural Style

## Overview

This is a TensorFlow implementation of the paper [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576) by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge.

<div align="center">
<img src="https://raw.githubusercontent.com/ftokarev/tf-neural-style/master/examples/input/tubingen.jpg" height="250px">
<img src="https://raw.githubusercontent.com/ftokarev/tf-neural-style/master/examples/output/tubingen-shipwreck.jpg" height="250px">

<img src="https://raw.githubusercontent.com/ftokarev/tf-neural-style/master/examples/output/tubingen-starry-night.jpg" height="250px">
<img src="https://raw.githubusercontent.com/ftokarev/tf-neural-style/master/examples/output/tubingen-the-scream.jpg" height="250px">

<img src="https://raw.githubusercontent.com/ftokarev/tf-neural-style/master/examples/output/tubingen-seated-nude.jpg" height="250px">
<img src="https://raw.githubusercontent.com/ftokarev/tf-neural-style/master/examples/output/tubingen-composition-vii.jpg" height="250px">
</div>

## Setup & Usage

    git clone https://github.com/ftokarev/tf-neural-style
    cd tf-neural-style
    virtualenv venv --python /usr/bin/python3
    source venv/bin/activate
    pip install -r requirements.txt
    cd model; ./get_model.sh; cd ..
    ./neural-style.py --content_image <path_to_content_img> --style_image <path_to_style_img>

## Acknowledgements

 - A lot of ideas are taken from Justin Johnson's implementation (https://github.com/jcjohnson/neural-style/)
 - VGG weights are from the "Deep Learning Models" project by François Chollet (https://github.com/fchollet/deep-learning-models)
