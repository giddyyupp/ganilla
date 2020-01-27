from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException

import os
import json
import urllib2
import sys
import time
import random

# adding path to geckodriver to the OS environment variable
os.environ["PATH"] += os.pathsep + os.getcwd()
download_path = "./down/" # where to download images.


def randdelay(a,b):
    time.sleep(random.uniform(a,b))


def main():
    searchtext = "miyazaki wallpaper"
    num_requested = int(5000)
    number_of_scrolls = num_requested / 400 + 1
    # number_of_scrolls * 400 images will be opened in the browser

    if not os.path.exists(download_path + searchtext.replace(" ", "_")):
        os.makedirs(download_path + searchtext.replace(" ", "_"))

    url = "https://www.google.co.in/search?q=" + searchtext + "&source=lnms&tbm=isch"
    # driver = webdriver.Chrome(r"D:\chromedriver")
    driver = webdriver.Firefox()
    driver.get(url)

    headers = {}
    headers['User-Agent'] = "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36"
    extensions = {"jpg", 'png', 'jpeg', 'tiff'}
    img_count = 0
    downloaded_img_count = 0

    for _ in xrange(number_of_scrolls):
        for __ in xrange(10):
            # multiple scrolls needed to show all 400 images
            driver.execute_script("window.scrollBy(0, 1000000)")
            randdelay(1, 3)
        # to load next 400 images
        randdelay(1, 3)
        try:
            driver.find_element_by_xpath("//input[@id='smb']").click()
        except Exception as e:
            print "Less images found:", e
            break

    # imges = driver.find_elements_by_xpath('//div[@class="rg_meta"]') # not working anymore
    imges = driver.find_elements_by_xpath('//div[contains(@class,"rg_meta")]')
    print "Total images:", len(imges), "\n"
    for img in imges:
        img_count += 1
        img_url = json.loads(img.get_attribute('innerHTML'))["ou"]
        img_type = json.loads(img.get_attribute('innerHTML'))["ity"]
        print "Downloading image", img_count, ": ", img_url
        try:
            # Save the window opener (current window)
            main_window = driver.current_window_handle

            scripttt = '''window.open('{link}')'''.format(link=img_url)
            driver.execute_script(scripttt)

            windows = driver.window_handles
            driver.switch_to.window(windows[1])

            delay = 10  # seconds
            try:
                img = WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.TAG_NAME, 'img')))
                print "Page is ready!"
            except TimeoutException:
                print "Loading took too much time!"

            # img = driver.find_element_by_tag_name('img')
            randdelay(2, 4) # wait some to get ss. eger cok fazla siyah resim olursa bunu arttirmaniz lazim
            img.screenshot(download_path + searchtext.replace(" ", "_") + "/" + str(downloaded_img_count) + ".png")
            randdelay(3, 5) # wait some to get ss. eger cok fazla siyah resim olursa bunu arttirmaniz lazim

            # Close current window
            driver.close()
            driver.switch_to.window(main_window)

            downloaded_img_count += 1
        except Exception as e:
            # Close current window
            driver.close()
            windows = driver.window_handles
            # get back to main window!!!
            driver.switch_to.window(windows[0])
            print "Download failed:", e
        finally:
            print
        if downloaded_img_count >= num_requested:
            break

    print "Total downloaded: ", downloaded_img_count, "/", img_count
    driver.quit()


if __name__ == "__main__":
    main()
