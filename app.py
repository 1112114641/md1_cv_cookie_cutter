"""
Create streamlit app to showcase model inference.

To run:
Follow `readme.MD` first steps, then:
```bash
streamlit run app.py
```
"""
# import pandas as pd
# import streamlit as st
# import wget
# import numpy as np
# from PIL import Image
# from src.run_inference import run_predict
# import pathlib
# import os
# import matplotlib.pyplot as plt

# os.chdir(str(pathlib.Path(__file__).parent.resolve()).split("/hush")[0])

# st.set_page_config(layout="wide")
# st.title("md1 CV Cookie Cutter")
# # st.image('image.png', use_column_width=True)
# st.subheader("Proof of Concept: Inference run on Docker container")
# st.markdown("")
# st.write(
#     "Please enter a link to a .flac/.mp3 soundfile for sound type classification in the sidebar."
# )
# st.sidebar.header("File downloading options:")
# link = st.sidebar.text_input("Please enter link to an flac/mp3 soundfile:")
# sample_rate = st.sidebar.radio("Sample Rate (Hz):", (44100, 8000, 4000), index=1)

# file = st.sidebar.file_uploader(
#     "Upload 44.1kHz soundfile of choice:", type=["flac", "mp3"]
# )

# begin = st.sidebar.button("Begin processing.")

# if begin and link:
#     # if ".flac" not in link:
#     #     out_path = "data/sfile.flac"
#     # else:
#     #     out_path = "data/"
#     file = wget.download(url=link, out="data")
#     sample, sample_rate = librosa.load(file, sr=int(sample_rate))
#     os.remove(file)
#     S = librosa.feature.melspectrogram(sample, sr=sample_rate, n_mels=128)
#     log_S = librosa.power_to_db(S, ref=np.max)
#     log_S = 255 * np.abs((log_S + np.abs(min(log_S.flatten()))) / min(log_S.flatten()))
#     img = Image.fromarray(log_S.astype(np.uint8)).resize((2671, 128))
#     new_file = os.path.splitext(file)[0] + ".jpg"
#     image = librosa.display.specshow(log_S, sr=sample_rate, x_axis="s", y_axis="mel")
#     file_name = os.path.split(new_file)[-1]
#     img.save(new_file)

#     plt.title("Mel Power Spectrogram")
#     plt.savefig(new_file + "_disp.jpg", facecolor="white")

#     preds = run_predict(image_file_path=os.path.abspath(new_file))

#     results = preds
#     results = pd.DataFrame.from_dict(results, orient="index", columns=["scores"])
#     results = results.sort_values(by="scores", ascending=False)

#     col1, col2 = st.columns(2)
#     col1.header("Spectrogram")
#     col1.image(new_file + "_disp.jpg", use_column_width=True)

#     col2.header("Model Results:")
#     col2.dataframe(results)
#     st.markdown("Please remember that these results are scores, not probabilities.")

#     os.remove(new_file + "_disp.jpg")
#     os.remove(new_file)
#     file = st.sidebar.empty()
#     link = st.sidebar.empty()

# elif begin and file:
#     data = file
#     file = "data/out.flac"
#     with open(file, "wb") as outfile:
#         outfile.write(data.getbuffer())
#     sample, sample_rate = librosa.load(file, sr=int(sample_rate))
#     os.remove(file)
#     S = librosa.feature.melspectrogram(sample, sr=sample_rate, n_mels=128)
#     log_S = librosa.power_to_db(S, ref=np.max)
#     log_S = 255 * np.abs((log_S + np.abs(min(log_S.flatten()))) / min(log_S.flatten()))
#     img = Image.fromarray(log_S.astype(np.uint8)).resize((2671, 128))
#     new_file = os.path.splitext(file)[0] + ".jpg"
#     image = librosa.display.specshow(log_S, sr=sample_rate, x_axis="s", y_axis="mel")
#     file_name = os.path.split(new_file)[-1]
#     img.save(new_file)

#     plt.title("Mel Power Spectrogram")
#     plt.savefig(new_file + "_disp.jpg", facecolor="white")

#     preds = run_predict(image_file_path=os.path.abspath(new_file))

#     results = preds
#     results = pd.DataFrame.from_dict(results, orient="index", columns=["scores"])
#     results = results.sort_values(by="scores", ascending=False)

#     col1, col2 = st.columns(2)
#     col1.header("Spectrogram")
#     col1.image(new_file + "_disp.jpg", use_column_width=True)

#     col2.header("Model Results:")
#     col2.dataframe(results)
#     st.markdown("Please remember that these results are scores, not probabilities.")

#     os.remove(new_file + "_disp.jpg")
#     os.remove(new_file)
#     file = st.sidebar.empty()
#     link = st.sidebar.empty()
