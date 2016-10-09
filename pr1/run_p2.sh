#exercise 2
for i in `seq 10 2 200`;
    do
        rm  cache/*.dat
        rm  cache/*.feat
        cd cache
        rm **/*.bovw
        cd ..
        python lab1.py -nc $i
    done  
