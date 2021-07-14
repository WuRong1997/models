import xlrd
import xlwt

#读excel
filepath = '***.xls'
excel = xlrd.open_workbook(filepath)
table = excel.sheets()[0]
table = excel.sheet_by_name('sheet1')
for i in range(1, table.nrows):
  id = table.cell(i,2).value

# 写excel
excel = xlwt.Workbook(encoding='ascii)
sheet = excel.add_sheet('sheet1')
sheet.write(0,0,'value')
excel.save('***.xls')
                      
# 将csv文件转化为xlsx文件
from pandas import read_csv
f = open('***.csv')
data = read_csv(f)
data.to_excel('***.xlsx')
