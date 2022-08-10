import argparse
import os
import sys
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from networks.drn_seg import DRNSeg
from utils.tools import *
from utils.visualize import *

from distutils.command.upload import upload
import streamlit as st
import global_classifier


def global_classify(uploaded_file, crop=False):
  model = global_classifier.load_classifier("weights/global.pth", 0)
  prob = global_classifier.classify_fake(model, uploaded_file, not crop)
  return("Probability being modified by Photoshop FAL: {:.2f}%".format(prob*100))


def local_classify(uploaded_file, crop=True):
    dest_folder = 'out/'
    model_path = 'weights/local.pth'
    gpu_id = 0
    no_crop = not crop

    # Loading the model
    if torch.cuda.is_available():
        device = 'cuda:{}'.format(gpu_id)
    else:
        device = 'cpu'

    model = DRNSeg(2)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict['model'])
    model.to(device)
    model.eval()

    # Data preprocessing
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    im_w, im_h = Image.open(uploaded_file).size
    if no_crop:
        face = Image.open(uploaded_file).convert('RGB')
    else:
        faces = face_detection(uploaded_file, verbose=False)
    if len(faces) == 0:
        print("no face detected by dlib, exiting")
        sys.exit()
    face, box = faces[0]
    face = resize_shorter_side(face, 400)[0]
    face_tens = tf(face).to(device)

    # Warping field prediction
    with torch.no_grad():
        flow = model(face_tens.unsqueeze(0))[0].cpu().numpy()
        flow = np.transpose(flow, (1, 2, 0))
        h, w, _ = flow.shape

    # Undoing the warps
    modified = face.resize((w, h), Image.BICUBIC)
    modified_np = np.asarray(modified)
    reverse_np = warp(modified_np, flow)
    reverse = Image.fromarray(reverse_np)

   
    finput = os.path.join(dest_folder, 'cropped_input.jpg')
    fwarped = os.path.join(dest_folder, 'warped.jpg')
    fheat = os.path.join(dest_folder, 'heatmap.jpg')

    flow_magn = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)

    # Saving the results
    modified.save(finput, quality=90)
    reverse.save(fwarped, quality=90)
    save_heatmap_cv(modified_np, flow_magn, fheat)

    return {
        "heatmap": os.path.join(dest_folder, 'heatmap.jpg'),
        "warped": os.path.join(dest_folder, 'warped.jpg'),
        "cropped_input": os.path.join(dest_folder, 'cropped_input.jpg')
    }
    
def st_ui():
    uploaded_file = st.file_uploader(
        label="Upload image",
        type=["jpeg", "jpg", "png"],
        accept_multiple_files=False,
        help="Upload an image to analyze",
    )
    if uploaded_file is not None:
        res = local_classify(uploaded_file)
        text = global_classify(uploaded_file)
        st.title(text)
        st.image(res["heatmap"], caption="Heatmap")
        st.image(res["warped"], caption="Warped")
        st.image(res["cropped_input"], caption="Cropped input")


if __name__ == "__main__":
    st_ui()
