import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import time

movie_path = 'movie/'
img_path = 'image/'

def sidebar_parm():
    col1, col2 = st.sidebar.columns(2)
    button_run = col1.button('START')
    button_stop = col2.button('STOP')
    mode = st.sidebar.selectbox('モードの選択', ['use movie file', 'use webcam'])
    fps_val = st.sidebar.slider('フレームレート', 1, 100, 50)

    uploaded_mv_file = None
    if mode == 'use movie file':
        uploaded_mv_file = st.sidebar.file_uploader('動画アップロード', type='mp4')
        if uploaded_mv_file is not None:
            st.sidebar.video(uploaded_mv_file)
    
    uploaded_img_file = None
    uploaded_img_file = st.sidebar.file_uploader('背景用画像アップロード', type=['jpg', 'jpeg', 'png'])
    if uploaded_img_file is not None:
        st.sidebar.image(uploaded_img_file)

    return button_run, button_stop, mode, fps_val, uploaded_mv_file, uploaded_img_file

def read_img_movie(img_path, uploaded_img_file, movie_path, uploaded_mv_file):
    img_file_path = img_path + uploaded_img_file.name
    mv_file_path = None
    cap_file = None

    org_bd_image = cv2.imread(img_file_path)

    if mode == 'use movie file':
        mv_file_path = movie_path + uploaded_mv_file.name
        cap_file = cv2.VideoCapture(mv_file_path)
    else:
        cap_file = cv2.VideoCapture(0)
    
    return org_bd_image, cap_file

def create_virtual_bg(button_stop, org_bd_image, cap_file, mp_selfie_segmentation, mode, fps_val):
    image_container = st.empty()
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
        while cap_file.isOpened:
            success, image = cap_file.read()

            if not success:
                break

            if button_stop == True:
                break

            # 動画ファイル処理
            if mode == 'use movie file':
                image = cv2.resize(image , dsize=None, fx=0.2, fy=0.2)
            else:
                image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height = rgb_image.shape[0]
            width = rgb_image.shape[1]

            results = selfie_segmentation.process(rgb_image)

            condition = np.stack((results.segmentation_mask,)*3, axis=-1)>0.5
            
            bg_image = cv2.resize(org_bd_image, dsize=(width,height))
            rgb_bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)

            output_image = np.where(condition, rgb_image, rgb_bg_image)
            
            time.sleep(1/fps_val)

            image_container.image(output_image)

            

    cap_file.release()

    return 0

if __name__ =='__main__':
    st.sidebar.text('各種設定')
    button_run, button_stop, mode ,fps_val, uploaded_mv_file, uploaded_img_file = sidebar_parm()

    st.title('バーチャル背景動画作成アプリ')
    if button_run == True:
        if mode == 'use movie file' and uploaded_mv_file is None:
            st.text('動画ファイルをアップロードしてください')
        elif uploaded_img_file is None:
            st.text('背景用画像をアップロードしてください')
        else:
            mp_selfie_segmentation = mp.solutions.selfie_segmentation
            org_bd_image, cap_file = read_img_movie(img_path, uploaded_img_file, movie_path, uploaded_mv_file)
            create_virtual_bg(button_stop, org_bd_image, cap_file, mp_selfie_segmentation, mode, fps_val)
