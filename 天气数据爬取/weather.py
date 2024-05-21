# coding=utf-8
import requests
import re
import csv
 
class WeatherForecast(object):
    def __init__(self):
        self.url = 'https://tianqi.2345.com/Pc/GetHistory?areaInfo%5BareaId%5D=54511&areaInfo%5BareaType%5D=2&date%5Byear%5D={0}&date%5Bmonth%5D={1}'
        self.headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36 Edg/107.0.1418.24',
            'accept-encoding': 'gzip, deflate, br'
        }
        self.data_list=[]
 
    def get_content(self,url):
        res = requests.get(url=url,headers=self.headers)
        content = res.json()
        # print(content['data'])
        return content['data']
 
    def parse_data(self,content):
        result = re.compile(r'<td>(?P<date>.*?)</td>.*?<td style="color:#ff5040;">(?P<max>.*?)</td>'
                            r'.*?<td style="color:#3097fd;" >(?P<min>.*?)</td>.*?<td>(?P<weather>.*?)</td>'
                            r'.*?<td>(?P<cloud>.*?)</td>.*?<td><span class="history-aqi wea-aqi.*?>(?P<sky>.*?)</span></td>',
                            re.S)
        find_result = result.finditer(content)
        for it in find_result:
            data_dict=it.groupdict()
            # print(data_dict)
            self.data_list.append(data_dict)
        return self.data_list
 
    def write_csv(self,data_list):
        with open('./test.csv','w',newline="")as f:
            writer=csv.writer(f)
            writer.writerow(['日期','最高温度','最低温度','天气','风力风向','空气质量'])
            for i in data_list:
                writer.writerow(i.values())
                print(i.values())
 
    def run(self):
        for year in range(2021, 2024, 1):
            for month in range(1, 13, 1):
                url = self.url.format(year, month)
                print('正在爬取第{0}年{1}月的天气!'.format(year, month))
                content = self.get_content(url)
                data=self.parse_data(content)
                self.write_csv(data)
            print('全部爬取完毕!')
 
 
if __name__ == '__main__':
    weather = WeatherForecast()
    weather.run()
