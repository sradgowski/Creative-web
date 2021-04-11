import streamlit as st
from streamlit import config
import src.IQCCD as main
import src.home as intro
import src.ccapp as ccapp
import src.tcapp as tcapp
import src.IQTCD as techdoc
PAGES={'About US':intro, 'IBM Quantum Creative Challenge(app)':ccapp, 'IBM Quantum Creative Challenge(docs)':main,'IBM Quantum Technical Challenge(app)':tcapp, 'IBM Quantum Technical Challenge(docs)':techdoc}
def write_page(page):  # pylint: disable=redefined-outer-name
    return page.write()
def main():
    st.set_page_config(page_title='QC HACKS')
    st.sidebar.title("Projects")
    choice=st.sidebar.radio("Explore the Projects below ?",tuple(PAGES.keys()))
    if choice ==None:
        intro.write()
    else:
        page=PAGES[choice]
        with st.spinner(f"Loading {choice} ..."):
            write_page(page)
    
if __name__ == "__main__":
    main()
