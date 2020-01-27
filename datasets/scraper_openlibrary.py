from selenium import webdriver
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.common.keys import Keys
import selenium.webdriver.support.ui as ui
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains

import time,random,socket,unicodedata
import os
import re


def randdelay(a,b):
    time.sleep(random.uniform(a,b))


def u_to_s(uni):
    return unicodedata.normalize('NFKD',uni).encode('ascii','ignore')


class OpenLibHelper(object):
    
    def __init__(self, login, pw):
        profile = webdriver.FirefoxProfile()
        profile.set_preference("browser.download.folderList", 2)
        profile.set_preference("browser.download.manager.showWhenStarting", False)
        profile.set_preference("browser.download.dir", "./")
        profile.set_preference("browser.helperApps.neverAsk.saveToDisk", "image/jpeg")
        self.browser = webdriver.Firefox(firefox_profile=profile)
        self.browser.maximize_window()
        # self.browser.find_element_by_xpath('/html/body').send_keys(Keys.F11)
        self.browser.get("https://openlibrary.org/account/login")
        emailElem = self.browser.find_element_by_name('username')
        emailElem.send_keys(login)
        passwordElem = self.browser.find_element_by_name('password')
        passwordElem.send_keys(pw)
        passwordElem.send_keys(Keys.RETURN)
        randdelay(5, 10)

    def search_author(self, author, data_folder, book_list=[]):

        searchElem = self.browser.find_element_by_name('q')
        searchElem.clear()
        searchElem.send_keys(author)
        searchElem.send_keys(Keys.RETURN)
        # wait up to 10 seconds for the elements to become available
        randdelay(5, 7)

        ## search results
        search_res = self.browser.find_element_by_id("searchResults")
        elementList = search_res.find_elements_by_tag_name("li")

        # traverse results
        for i in range(len(elementList)):
            element = self.browser.find_element_by_id("searchResults").find_elements_by_tag_name("li")[i]
            try:

                # Save the window opener (current window)
                main_window = self.browser.current_window_handle

                # open result in new tab
                # borrow_button = element.find_element_by_xpath("//div[@class='searchResultItemCTA']//div[@class='searchResultItemCTA-lending']")
                borrow_button_first = element.find_element_by_tag_name('a')
                borrow_button_first.send_keys(Keys.CONTROL + Keys.RETURN)
                randdelay(5, 7) # wait for new tab to load

                # Get windows list and put focus on new window (which is on the 1st index in the list)
                windows = self.browser.window_handles
                self.browser.switch_to.window(windows[1])
                # do whatever you have to do on this page, we will just got to sleep for now
                randdelay(7, 11)

                title = self.browser.find_element_by_xpath(
                    "//body[@id='user']//div[@id='test-body-mobile']//div[@class='contentContainer']//div[@id='contentHead']//span[@itemprop='name']").text
                if (book_list) and (not title.lower() in book_list):
                    raise Exception('Book is not in the list!')

                dir_name = data_folder + "/" + title
                # if we already downloaded this book, move on!!
                if os.path.exists(dir_name):
                    raise Exception('Already downloaded this book!')  # Don't! If you catch, likely to hide bugs.
                else: # else lets create a dir and download!
                    os.makedirs(dir_name)

                #now click Borrow!! or maybe we already loaned this so try to read at least.
                try:
                    borrow_button_second = self.browser.find_element_by_link_text('Borrow')
                except:
                    # try:
                    borrow_button_second = self.browser.find_element_by_link_text('Read')
                    # except:
                    #     continue # no loans available so move to the next book.

                borrow_button_second.click()
                randdelay(20, 23)
                #switch to one page
                one_page = self.browser.find_element_by_xpath(
                    "//div[@id='IABookReaderWrapper']//div[@id='BookReader']//div[@class='BRfooter']//div[@class='BRnav BRnavDesktop']//div[@class='BRpage']//button[@bt-xtitle='One-page view']")
                one_page.click()

                # now take source text, we will use it to generate full URLs
                page_data = self.browser.page_source
                page_list = page_data.splitlines()
                # find the item index similar to this:
                # url: '//ia903101.us.archive.org/BookReader/BookReaderJSIA.php?id=lettherebelight0000unse&itemPath=
                # /5/items/lettherebelight0000unse&server=ia903101.us.archive.org&format=jsonp&subPrefix=
                # lettherebelight0000unse&requestUri=/stream/lettherebelight0000unse&version=mHe9koCz',
                str___ = page_list[267]  # this index may change if site is updated!

                aaa = str___.replace(" ", "") # remove heading whitespaces
                tokens_url = re.split(';|,|\.|/|=|&|', aaa) # now split into tokens
                # tk0 = tokens_url[0]
                # tk1 = tokens_url[1]
                tk2 = tokens_url[2] # this is similar to ia903101
                tk3 = tokens_url[3] # us
                tk4 = tokens_url[4] # archive
                tk5 = tokens_url[5] # org
                tk6 = tokens_url[6] # BookReader
                tk12 = tokens_url[12]
                tk13 = tokens_url[13]
                tk14 = tokens_url[14]

                # here turn pages till end!!!
                for j in range(1000): #adjust this to pages of a book
                    # print self.browser.current_url
                    # str2 = "https://ia800706.us.archive.org/BookReader/BookReaderImages.php?zip=/35/items/billpeetautobiog00peet/billpeetautobiog00peet_jp2.zip&file=billpeetautobiog00peet_jp2/billpeetautobiog00peet_0019.jp2&scale=1&rotate=0"
                    page_num = '0001'
                    if j < 9:
                        page_num = '000' + str(j+1)
                    elif j < 99:
                        page_num = '00' + str(j+1)
                    else:
                        page_num = '0' + str(j+1)

                    file_path = dir_name + "/page_{PAGE_NUM}.png".format(PAGE_NUM=page_num)

                    source_URL = "https://" + tk2 + '.' + tk3 + '.' + tk4 + '.' + \
                                 tk5 + '/' + tk6 + "/BookReaderImages.php?zip=/" + tk12 + "/" + \
                                 tk13 + "/" + tk14 + '/' + tk14 + "_jp2.zip&file=" + \
                                 tk14 + "_jp2/" + tk14 + "_{PAGE_NUM}".format(PAGE_NUM=page_num) + ".jp2&scale=1&rotate=0"

                    scripttt = '''window.open('{link}')'''.format(link=source_URL) #,'_blank'
                    self.browser.execute_script(scripttt)
                    # Get windows list and put focus on new window (which is on the 2nd index in the list)
                    windows = self.browser.window_handles
                    self.browser.switch_to.window(windows[2])
                    self.browser.maximize_window()

                    randdelay(15, 18) # wait for page to load

                    try:
                        img = self.browser.find_element_by_tag_name('img')
                        # now lets get full image!!!
                        self.browser.execute_script("arguments[0].setAttribute('class','overflowingVertical')", img)
                        self.browser.execute_script("arguments[0].removeAttribute('width','')", img)
                        self.browser.execute_script("arguments[0].removeAttribute('height','')", img)

                        img.screenshot(file_path)
                        randdelay(3, 5)
                    except:
                        # Close current window
                        self.browser.close()
                        # get back to prev window!!!
                        windows = self.browser.window_handles
                        self.browser.switch_to.window(windows[1])
                        break # break from the loop

                    # Close current window
                    self.browser.close()

                    # get back to prev window!!!
                    windows = self.browser.window_handles
                    self.browser.switch_to.window(windows[1])

                # now we can return our book!!
                self.browser.find_element_by_link_text("Return Book").click()
                randdelay(1, 2)
                # close float tab, and go back to main window
                try:
                    return_button = self.browser.find_element_by_xpath("//div[@id='colorbox']//div[@id='cboxWrapper']//div[@id='cboxContent']//div[@id='cboxLoadedContent']//div[@class='center BRfloat']//div[@class='center BRfloatFoot']//button[@class='action red']")
                    return_button.click()
                    randdelay(8, 11)
                    # Close current window
                    self.browser.close()
                    # Put focus back on main window
                    self.browser.switch_to.window(main_window)
                except:
                    print "no return button"

                randdelay(3, 5)
            except:
                # Close current window
                self.close()
                # Put focus back on main window
                self.browser.switch_to.window(main_window)
                continue

    def close(self):
        self.browser.close()
