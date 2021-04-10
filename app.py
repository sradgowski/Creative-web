import streamlit as st
import src.main as main
import src.home as intro
PAGES={'intro':intro, 'main':main}
def write_page(page):  # pylint: disable=redefined-outer-name
    return page.write()
def main():
    st.set_page_config(page_title='QC HACKS')
    st.sidebar.title("Projects")
    choice=st.sidebar.radio("Explore the Projects below ?",tuple(PAGES.keys()))
    if choice ==None:
        Intro.write()
    else:
        page=PAGES[choice]
        with st.spinner(f"Loading {choice} ..."):
            write_page(page)
    
if __name__ == "__main__":
    main()
