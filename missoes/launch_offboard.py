import subprocess

subprocess.run("""python3 offboard1.py & 
		python3 offboard2.py &
		python3 offboard3.py &
		python3 offboard4.py &
		python3 offboard5.py &
		python3 offboard6.py &
		python3 offboard7.py &
		python3 offboard8.py &
		python3 offboard9.py""", shell=False)
       
