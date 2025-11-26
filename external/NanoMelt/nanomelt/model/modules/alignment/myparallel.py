"""
 Copyright 2023. Aubin Ramon and Pietro Sormanni. CC BY-NC-SA 4.0
"""

try:
    from collections import Sequence  # older python
except Exception:
    from collections.abc import Sequence  #  python>= 3.10

import multiprocessing
import os
import pickle
import sys
import time
from traceback import print_exc

status = {"inited": False}


def __init():
    global status
    if not multiprocessing.get_start_method(allow_none=True):
        multiprocessing.set_start_method("fork", force=True)
    # Warning The 'spawn' and 'forkserver' start methods
    # cannot currently be used with “frozen” executables (i.e., binaries produced
    # by packages like PyInstaller and cx_Freeze) on Unix. The 'fork' start method does work.
    # spawn does not work in example in help of  pool_function, fork does.
    manager = multiprocessing.Manager()
    status[
        "data_queue"
    ] = (
        manager.Queue()
    )  # set up a queue to retrieve the results from the children processes
    status["inited"] = True


def pool_worker(args):
    """
    aux of pool_function
    # args[0] is the function, args[1] its arguments and args[2] an index corresponding to input bit
    #  **args[3] can be kargs if given
    """
    # print 'args=',args
    fun_args = args[1]
    if not isinstance(fun_args, Sequence) or type(fun_args) is str:
        fun_args = [fun_args]
    if len(args) > 3:
        status["data_queue"].put((args[0](*fun_args, **args[3]), args[2]))
    else:
        status["data_queue"].put((args[0](*fun_args), args[2]))

def pool_worker_chuncked(args):
    """
    aux of pool_function
    # args[0] is the function, args[1] its arguments and args[2] an index corresponding to input bit
    #  **args[3] can be kargs if given
    """
    # print 'args=',args
    fun_args = args[1]
    #if not isinstance(fun_args, Sequence) or type(fun_args) is str: # this must be list of lists in pool_worker_chuncked
    #    fun_args = [fun_args]
    if len(args) > 3:
        status["data_queue"].put((args[0](fun_args, **args[3]), args[2]))
    else:
        status["data_queue"].put((args[0](fun_args), args[2]))


def pool_function(function, iterable_of_args, ncpu=None, chunksize=None, **kwargs):
    """
        to run a function parallel
        return results a dict whose keys are index and values results of corresponding iterable_of_args
            the results dict will be sorted in order of completion of each pool, so to iterate on it is better to use
             for j,x in enumerate(iterable_of_args) :
                res=results[j]
            so to get the results in the order of the input iterable_of_args
        ** kwargs will be the same for all calls
        the chunksize parameter will cause the iterable to be split into pieces of approximately that size, and each piece is submitted as a separate task.
        if chunksize=='auto' : chunksize = len(iterable_of_args) // ncpu 
        example:
import myparallel
import time
def fun(x, exponent=2): # it just sleeps for one second and returns the exponent power of the variable
    time.sleep(1)
    return x**exponent

chunksize=None # if you give chunksize=10 it should take about 10 s on 2 cores (for the input range(0,20))
inputs=range(0,20)
sta=time.time()
results= myparallel.pool_function(fun,inputs, exponent=2,chunksize=chunksize) # pass relevant kwargs that will be the same for all
took=time.time()-sta
print('took',took)
        """
    if not status["inited"]:
        __init()  # init the multiprocessing environment
    avail_cpu = multiprocessing.cpu_count()
    if ncpu is None:
        ncpu = avail_cpu
    else:
        ncpu = min([ncpu, avail_cpu])  # sentinel, don't assign more cpus than available
    if chunksize=='auto' :
        chunksize = len(iterable_of_args) // ncpu
    try:
        pool = multiprocessing.Pool(ncpu)
        if len(kwargs) > 0:
            pool.map(
                pool_worker,
                list(
                    zip(
                        [function] * len(iterable_of_args),
                        iterable_of_args,
                        list(range(len(iterable_of_args))),
                        [kwargs] * len(iterable_of_args),
                    )
                ),
                chunksize=chunksize,
            )
        else:
            pool.map(
                pool_worker,
                list(
                    zip(
                        [function] * len(iterable_of_args),
                        iterable_of_args,
                        list(range(len(iterable_of_args))),
                    )
                ),
                chunksize=chunksize,
            )
        # pool.terminate()
    except:
        pool.close()
        raise
    pool.close()  # shut down the pool - otherwise too many files opened can happen
    # retrieve output
    i = 0
    results = {}
    while i < len(iterable_of_args):
        out_child, out_index = status["data_queue"].get()
        results[out_index] = out_child
        i += 1
    return results


def pool_on_chunks(function, iterable_of_args, ncpu=None,**kwargs) :
    '''
    expects the function to get a list of args as input (e.g. the function loops on a list to fill something that it then returns)
      such list will be split in chunks and then the function will run in parallel on each chunk.
    return results a dict whose keys are index (of each chunk) and values results of function as run on each chunk
            the results dict will be sorted in order of completion of each pool, so to 
            reconstruct the orignial order
             for j in sorted(results) : # or in range(0,len(results))
                
    '''
    list_of_lists = chunk_list_of_args(iterable_of_args, ncpu)
    if not status["inited"]:
        __init()  # init the multiprocessing environment
    try:
        pool = multiprocessing.Pool(ncpu)
        if len(kwargs) > 0:
            pool.map(
                pool_worker_chuncked,
                list(
                    zip(
                        [function] * len(list_of_lists),
                        list_of_lists,
                        list(range(len(list_of_lists))),
                        [kwargs] * len(list_of_lists),
                    )
                ),
                chunksize=1)
        else:
            pool.map(
                pool_worker_chuncked,
                list(
                    zip(
                        [function] * len(list_of_lists),
                        list_of_lists,
                        list(range(len(list_of_lists))),
                    )
                ),
                chunksize=1)
        # pool.terminate()
    except:
        pool.close()
        raise
    pool.close()  # shut down the pool - otherwise too many files opened can happen
    # retrieve output
    i = 0
    results = {}
    while i < len(list_of_lists):
        out_child, out_index = status["data_queue"].get()
        results[out_index] = out_child
        i += 1
    return results


def chunk_list_of_args(iterable_of_args, ncpu):
    avail_cpu = multiprocessing.cpu_count()
    if ncpu is None:
        ncpu = avail_cpu
    args_per_cpu = len(iterable_of_args) // ncpu
    list_of_lists = [
        iterable_of_args[j * args_per_cpu : j * args_per_cpu + args_per_cpu]
        for j in range(ncpu)
    ]
    for j, fil in enumerate(
        iterable_of_args[(ncpu - 1) * args_per_cpu + args_per_cpu :]
    ):  # add remaining
        list_of_lists[j] += [fil]
    print(
        "myparallel.chunk_list_of_args on %d cpus with %d chunks of sizes %s type %s\n"
        % (
            ncpu,
            len(list_of_lists),
            str([len(x) for x in list_of_lists]),
            str(type(list_of_lists)),
        )
    )
    return list_of_lists


def run_parallel(function, iterable_of_args, ncpu=None, **kwargs):
    """
Typically used pool_function
        to run a function parallel
        return results a dict whose keys are index and values results of corresponding iterable_of_args
        ** kwargs will be the same for all calls
        example (see also chunk_list_of_args to make suitable chunks):
import myparallel
from time import sleep
def fun(x, exponent=2): # it just sleeps for one second and returns the exponent power of the variable
    sleep(1)
    return x**exponent

inputs=range(0,20)
results= myparallel.run_parallel(fun,inputs, exponent=2) # pass relevant kwargs that will be the same for all
        """
    if not status["inited"]:
        __init()  # init the multiprocessing environment
    avail_cpu = multiprocessing.cpu_count()
    if ncpu is None:
        ncpu = avail_cpu
    else:
        if ncpu > avail_cpu:
            sys.stderr.write(
                "**WARNING** in myparallel ncpu>avail_cpu %d and %d setting to avail_cpu\n"
                % (ncpu, avail_cpu)
            )
        ncpu = min([ncpu, avail_cpu])  # sentinel, don't assign more cpus than available
    try:
        # elegant chunks but assumes function would take as input a list, which may not be the general case
        # args_per_cpu=len(iterable_of_args)//ncpu
        # list_of_lists=[ iterable_of_args[j*args_per_cpu:j*args_per_cpu+args_per_cpu] for j in range(ncpu) ]
        # for j, fil in enumerate(iterable_of_args[(ncpu-1)*args_per_cpu+args_per_cpu:]) : # add remaining
        #    list_of_lists[j]+=[fil]
        # print ("Attemptnig parallel run on %d cpus with %d chunks of sizes %s type %s\n" % (ncpu,len(list_of_lists),str([len(x) for x in list_of_lists]),str(type(list_of_lists)) ) )
        function_to_pull = lambda *args: function(*args, **kwargs)
        pool = multiprocessing.Pool(ncpu)
        pool.map(
            pool_worker,
            list(
                zip(
                    [function_to_pull] * len(iterable_of_args),
                    iterable_of_args,
                    list(range(len(iterable_of_args))),
                )
            ),
        )
        # pool.terminate()
    except:
        pool.close()
        raise
    pool.close()  # shut down the pool - otherwise too many files opened can happen
    # retrieve output
    i = 0
    results = {}
    while i < len(iterable_of_args):
        out_child, out_index = status["data_queue"].get()
        results[out_index] = out_child
        i += 1
    return results
