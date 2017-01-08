# coding=utf-8
"""
@ author: cer
spider to crawl names
"""

import requests
from lxml import etree
import cPickle as pickle
import re
import os
import time
import random


def load_base_data():
    """加载基础页的数据"""
    base_data = "data/base_data.pkl"
    # 有则加载
    if os.path.exists(base_data):
        with open(base_data) as f:
            name_base_data_pkl = pickle.load(f)
        return name_base_data_pkl
    # 无则爬取
    base_url = "http://www.resgain.net/xmdq.html"
    r = requests.get(base_url)
    # print r.text
    tree = etree.HTML(r.text)
    name_links = tree.xpath("//div[@class='container']/div[@class='row']/div[@class='col-xs-12']/a[@class='btn btn2']")
    name_base_data = []
    p_xing = re.compile(u"[\u4e00-\u9fa5]{1,2}姓")
    p_num = re.compile(u"[\d]+")
    for name_l in name_links:
        link = name_l.xpath("./@href")[0]
        title = name_l.xpath("./@title")[0]
        print title
        datum = {}
        # print len(name_links)
        # print "link: ", link, "title: ", title
        s_xing = p_xing.search(title)
        if not s_xing:
            continue
        xing = s_xing.group(0)[:-1]
        s_num = p_num.search(title)
        num = s_num.group(0)
        datum["xing"] = xing
        datum["link"] = link
        datum["title"] = title
        datum["num"] = int(num)
        name_base_data.append(datum)
    # 保存到pkl
    with open(base_data, "wb") as base_file:
        # Pickle the list using the highest protocol available.
        pickle.dump(name_base_data, base_file, -1)
    return name_base_data


def fix_base_data():
    """修改基础页有问题的数据"""
    base_data = "data/base_data.pkl"
    # 有则加载
    if os.path.exists(base_data):
        with open(base_data) as f:
            data = pickle.load(f)
            data[366]["link"] = data[366]["link"].replace(")", "")
            print data[366]
        with open(base_data, "wb") as base_file:
            # Pickle the list using the highest protocol available.
            pickle.dump(data, base_file, -1)


def load_one_xing(one_base):
    """爬取单个姓的所有名字"""
    one_pkl_name = "data/"+one_base["xing"]+".pkl"
    datum = {}
    links = list()
    links.append(one_base["link"].replace("name_list.html", "name/boys.html"))
    links.append(one_base["link"].replace("name_list.html", "name/girls.html"))
    # 目前每个姓，每个性别，只能爬取300个名字
    for i in range(2, 11):
        links.append(one_base["link"].replace("name_list.html", "name/boys_{}.html".format(i)))
        links.append(one_base["link"].replace("name_list.html", "name/girls_{}.html".format(i)))
    print links
    names = []
    for link in links:
        print "crawling {} ... ".format(link)
        r = requests.get(link)
        tree = etree.HTML(r.text)
        name_links = tree.xpath("//div[@class='container']/div[@class='row']/div[@class='col-xs-12']/a[@class='btn btn-link']")
        for name_l in name_links:
            names.append(name_l.xpath("./text()")[0])
            # 随机等待1-5秒
        sec = random.randint(1, 3)
        print "waiting for {} secs".format(sec)
        # time.sleep(sec)
    datum["xing"] = one_base["xing"]
    datum["names"] = names
    datum["num"] = len(names)
    # print r.text
    with open(one_pkl_name, "wb") as detail_file:
        # Pickle the list using the highest protocol available.
        pickle.dump(datum, detail_file, -1)
    #return datum


def load_all_details(base_data):
    """获取所有姓的数据，有则加载pkl，无则爬取"""
    for i, one_base in enumerate(base_data):
        if i == 2:
            print "crawling no.{} of total {} xing".format(i + 1, len(base_data))
            load_one_xing(one_base)


def collect_all_pkls(base_data):
    """把所有的姓pkl收集并合并保存在统一的pkl"""
    detail_data_name = "data/detail_data.pkl"
    detail_data = []
    num_total = 0
    xing_total = 0
    for one_data in base_data:
        one_pkl_n = "data/" + one_data["xing"] + ".pkl"
        if os.path.exists(one_pkl_n):
            with open(one_pkl_n) as f:
                one_detail = pickle.load(f)
                num_total += one_detail["num"]
                xing_total += 1
                detail_data.append(one_detail)
        else:
            print "lack data: ", one_data["xing"]
    print "total num of xing types: ", xing_total
    print "total name nums:", num_total
    # 保存到pkl
    with open(detail_data_name, "wb") as detail_file:
        # Pickle the list using the highest protocol available.
        pickle.dump(detail_data, detail_file, -1)

if __name__ == '__main__':
    base_data = load_base_data()
    # fix_base_data()
    # print len(base_data)
    # print base_data[:5]
    # load_all_details(base_data)
    collect_all_pkls(base_data)



