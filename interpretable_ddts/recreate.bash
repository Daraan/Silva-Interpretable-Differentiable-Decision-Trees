# Execute with bash not sh
#!/bin/bash
source env/bin/activate
cd runfiles

# NO GPU
# MLP
processes=0
for rl_env in "lunar" "cart"
do
    for i in 0 1 2 4 8 16 32
    do
        echo "Running MLP layers: iteration $i: $rl_env"
        # Add your commands here
        start=`date +%s`
        python gym_runner.py -a mlp -e 1000 -env $rl_env --num_hidden $i --process_number $processes $1 &
        end=`date +%s`
        runtime=$((end-start))
        echo "Runtime: $runtime"
        # 25 cores
        processes=$(($processes + 1))
        if (( (($processes % 5)) == 0 )); then
            echo "Waiting for processes to finish"
            wait
        fi
    done
done


# DDT
for rl_env in "lunar" "cart"
do
    for i in 2 4 8 16 32
    do
        echo "Running DDT iteration $i: $rl_env"
        # Add your commands here
        start=`date +%s`
        python gym_runner.py -a ddt -e 1000 -env $rl_env --num_leaves $i --process_number $processes $1 &
        end=`date +%s`
        echo "Runtime: $runtime"
        # 25 cores
        processes=$(($processes + 1))
        if (( (($processes % 5)) == 0 )); then
            echo "Waiting for processes to finish"
            wait
        fi
    done
done


# DDT rule list
for rl_env in "lunar" "cart"
do
    for i in 2 4 8 16 32
    do
        echo "Running DDT Rule list iteration $i: $rl_env"
        # Add your commands here
        start=`date +%s`
        python gym_runner.py -a ddt -e 1000 -env $rl_env --num_leaves $i  --rule_list --process_number $processes $1 &
        echo "proceeses $processes"
        end=`date +%s`
        echo "Runtime: $runtime"
        # 25 cores
        processes=$(($processes + 1))
        if (( (($processes % 5)) == 0 )); then
            echo "Waiting for processes to finish"
            wait
        fi
    done
done


# Discrete
python run_discrete_agent.py -env cart -f -r -d --all
python run_discrete_agent.py -env lunar -f -r -d --all