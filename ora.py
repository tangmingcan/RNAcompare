import time
from jupyter_client import KernelManager

def main():
    # Create a new kernel manager
    km = KernelManager(kernel_name='python3')
    km.start_kernel()

    # Create a client to interact with the kernel
    kc = km.client()
    kc.start_channels()

    # Ensure the client is connected before executing code
    kc.wait_for_ready()

    # Execute the code
    code = '''
import decoupler as dc
import scanpy as sc
import pandas as pd
adata1 = sc.read("ora_12345.h5ad")
msigdb1 = pd.read_csv('msigdb.csv')
msigdb1 = msigdb1[msigdb1['collection']=='reactome_pathways']
msigdb1 = msigdb1[~msigdb1.duplicated(['geneset', 'genesymbol'])]
dc.run_ora(
    mat=adata1,
    net=msigdb1,
    source='geneset',
    target='genesymbol',
    verbose=True,
    use_raw=False
)
adata1.obsm['ora_estimate'].to_json(orient='split')
'''
    msg_id = kc.execute(code)

    # Wait for the result and display it
    while True:
        try:
            msg = kc.get_iopub_msg(timeout=2)
            content = msg["content"]
            print(msg['msg_type'])
            print(msg)

            # When a message with the text stream comes and it's the result of our execution
            if msg["msg_type"] == "execute_result":
                print(content)
                break
        except KeyboardInterrupt:
            print("Interrupted by user.")
            break
        except:
            # If no messages are available, we'll end up here, but we can just continue and try again.
            pass

    # Cleanup
    kc.stop_channels()
    km.shutdown_kernel()

if __name__ == '__main__':
    main()
