
import sys
import time
def process_bar(precent, width=50):
    use_num = int(precent*width)
    space_num = int(width-use_num)
    precent = precent*100
    # print('[%s%s]%d%%'%(use_num*'#', space_num*' ',precent), end='\r')
    print('[%s%s]%d%%'%(use_num*'#', space_num*' ',precent),file=sys.stdout,flush=True, end='\r')