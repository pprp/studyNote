import time
def get_time(func):
    startTime = time.time()
    func()
    endTime = time.time()
    processTime = (endTime - startTime) * 1000
    print ("The function timing is %f ms" %processTime)
    
@get_time
def myfunc():
	print("start")
	time.sleep(0.8)
	print("end")

if __name__ == "__main__":
    myfunc