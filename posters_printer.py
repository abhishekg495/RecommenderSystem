import streamlit as st
import os
import base64


class posters_printer:
    def __init__(self):
        pass

    @st.cache(allow_output_mutation=True)
    def get_base64_of_bin_file(self, bin_file):
        with open(bin_file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()

    ######## GENERATE HTML CODE FOR AN IMAGE WITH HYPERLINK ####################################
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
                <p style="display:inline-block;font-size:15px;font-weight:100;width:100%">"""
            + image_caption
            + f"""
                </p>
            </div>"""
        )
        return html_code

    ##############################################################################################

    def print(self, rec):
        if len(rec) == 0:
            st.write(" ")
            st.write(" ")
            cols = st.columns(3)
            cols[1].image(
                "Posters/empty.png",
                width=270,
                caption="Ralph couldn't find anything for your current search. Try another search maybe ?",
            )
        else:
            columns = st.columns(5)  ## no. of columns to split posters into
            for movie in range(len(rec)):
                try:
                    img_src = (
                        "Posters/" + str(rec.index[movie]) + ".jpg"
                    )  ## get movie poster by movie ID
                except:
                    img_src = "Posters/unavailable.png"  ## get default image if movie poster not in folder

                img_html = self.get_img_with_href(
                    img_src,
                    rec.iloc[movie]["title"],
                    rec.iloc[movie]["imdb_link"],
                )
                columns[movie % len(columns)].markdown(img_html, unsafe_allow_html=True)
