import streamlit as st
import os
import base64


class posters_grid:
    def __init__(self):
        pass

    @st.cache(allow_output_mutation=True)
    def get_base64_of_bin_file(self, bin_file):
        with open(bin_file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()

    @st.cache(allow_output_mutation=True)
    def get_img_with_href(self, local_img_path, image_caption, target_url):
        img_format = os.path.splitext(local_img_path)[-1].replace(".", "")
        bin_str = self.get_base64_of_bin_file(local_img_path)
        html_code = (
            f"""
            <div style="display:float;justify-content:center;text-align:center;">
                <a href="{target_url}">
                    <img src="data:image/{img_format};base64,{bin_str}" style="width:100%;" />
                </a>
                <p style="display:inline-block;font-size:1vw;font-weight:100;width:100%">"""
            + image_caption
            + f"""
                </p>
            </div>"""
        )
        return html_code

    def print(self, rec):
        if len(rec) == 0:
            st.image("Posters/empty.png", width=300)
        else:
            columns = st.columns(5)
            for movie in range(len(rec)):
                try:
                    img_src = "Posters/" + str(rec.index[movie]) + ".jpg"
                    img_html = self.get_img_with_href(
                        img_src,
                        rec.iloc[movie]["title"],
                        rec.iloc[movie]["imdb_link"],
                    )
                    columns[movie % len(columns)].markdown(
                        img_html, unsafe_allow_html=True
                    )
                except:
                    img_src = "Posters/unavailable.png"
                    img_html = get_img_with_href(
                        img_src,
                        rec.iloc[movie]["title"],
                        rec.iloc[movie]["imdb_link"],
                    )
                    columns[movie % len(columns)].markdown(
                        img_html, unsafe_allow_html=True
                    )
