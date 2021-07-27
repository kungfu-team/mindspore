import os


def parse_kungfu_env():
    self = os.getenv('KUNGFU_SELF_SPEC')
    peers = os.getenv('KUNGFU_INIT_PEERS').split(',')
    rank = peers.index(self)

    #print(peers)
    #print(self)

    return {
        'rank': rank,
        'size': len(peers),
    }
