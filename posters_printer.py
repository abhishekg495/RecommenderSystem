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
        html_code = f"""
            <div class="movie-poster">
                <a href="{target_url}">
                    <img src="data:image/{img_format};base64,{bin_str}" style="width:100%;"/>
                </a>
            </div>"""
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
            columns = st.columns(
                [1, 1, 1, 1, 1]
            )  ## no. of columns to split posters into
            for movie in range(len(rec)):
                movie_name = rec.iloc[movie]["title"]
                movie_link = rec.iloc[movie]["imdb_link"]
                col = columns[movie % len(columns)]
                try:
                    img_src = (
                        "Posters/" + str(rec.index[movie]) + ".jpg"
                    )  ## get movie poster by movie ID
                except:
                    img_src = "Posters/unavailable.png"  ## get default image if movie poster not in folder

                img_html = self.get_img_with_href(
                    img_src,
                    movie_name,
                    movie_link,
                )
                col.markdown(img_html, unsafe_allow_html=True)

                ### ADD/REMOVE OPTIONS FOR WATCHLIST #####################################
                if (movie_name) not in st.session_state["watchlist"].movies_list:
                    add_movie = col.button(
                        "Add to watchlist",
                        on_click=st.session_state["watchlist"].add,
                        args=[(movie_name)],
                        key=movie,
                    )
                else:
                    remove_movie = col.button(
                        "Remove",
                        on_click=st.session_state["watchlist"].remove,
                        args=[(movie_name)],
                        key=movie,
                    )
                ################################################################################
