#exercise 2
for i in `seq 50 5 150`;
    do
        rm  cache/*.dat
        rm  cache/*.feat
        cd cache
        rm **/*.bovw
        cd ..
        python lab1.py -c $i -t 4
    done  
