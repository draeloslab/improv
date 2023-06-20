import os
import psutil
import pytest
import subprocess
from improv.link import Link  # , AsyncQueue
from improv.actor import AbstractActor as Actor
from improv.store import Store
from demos.basic.actors.zmqActor import ZmqPSActor, ZmqRRActor

import traceback
import asyncio
# set global_variables

pytest.example_string_links = {}
pytest.example_links = {}


@pytest.fixture()
def setup_store(scope="module"):
    """Fixture to set up the store subprocess with 10 mb."""

    p = subprocess.Popen(
        ["plasma_store", "-s", "/tmp/store", "-m", str(10000000)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    yield p
    p.kill()
    p.wait()


@pytest.fixture()
def init_actor():
    """Fixture to initialize and teardown an instance of actor."""

    act = Actor("Test")
    yield act
    act = None


@pytest.fixture()
def example_string_links():
    """Fixture to provide a commonly used test input."""

    pytest.example_string_links = {"1": "one", "2": "two", "3": "three"}
    return pytest.example_string_links


@pytest.fixture()
def example_links(setup_store):
    """Fixture to provide link objects as test input and setup store."""
    Store(store_loc="/tmp/store")

    acts = [Actor("act" + str(i)) for i in range(1, 5)]  # range must be even

    links = [
        Link("L" + str(i + 1), acts[i], acts[i + 1]) for i in range(len(acts) // 2)
    ]
    link_dict = {links[i].name: links[i] for i, l in enumerate(links)}
    pytest.example_links = link_dict
    return pytest.example_links


@pytest.mark.parametrize(
    ("attribute", "expected"),
    [
        ("q_watchout", None),
        ("name", "Test"),
        ("links", {}),
        ("lower_priority", False),
        ("q_in", None),
        ("q_out", None),
    ],
)
def test_default_init(attribute, expected, init_actor):
    """Tests if the default init attributes are as expected."""

    atr = getattr(init_actor, attribute)

    assert atr == expected


def test_repr_default_initialization(init_actor):
    """Test if the actor representation has the right dict keys."""

    act = init_actor
    rep = act.__repr__()
    assert rep == "Test: dict_keys([])"


def test_repr(example_string_links):
    """Test if the actor representation has the right, nonempty, dict."""

    act = Actor("Test")
    act.setLinks(example_string_links)
    assert act.__repr__() == "Test: dict_keys(['1', '2', '3'])"


def test_setStore(setup_store):
    """Tests if the store is started and linked with the actor."""

    act = Actor("Acquirer")
    store = Store(store_loc="/tmp/store")
    act.setStore(store.client)
    assert act.client is store.client


@pytest.mark.parametrize(
    "links", [(pytest.example_string_links), ({}),
              (pytest.example_links), (None)]
)
def test_setLinks(links):
    """Tests if the actors links can be set to certain values."""

    act = Actor("test")
    act.setLinks(links)
    assert act.links == links


@pytest.mark.parametrize(
    ("qc", "qs"),
    [
        ("comm", "sig"),
        (None, None),
        ("", ""),
        ("LINK", "LINK"),  # these are placeholder names (store is not setup)
    ],
)
def test_setCommLinks(example_links, qc, qs, init_actor, setup_store):
    """Tests if commLinks can be added to the actor"s links."""

    if qc == "LINK" and qs == "LINK":
        qc = Link("L1", Actor("1"), Actor("2"))
        qs = Link("L2", Actor("3"), Actor("4"))
    act = init_actor
    act.setLinks(example_links)
    act.setCommLinks(qc, qs)

    example_links.update({"q_comm": qc, "q_sig": qs})
    assert act.links == example_links


@pytest.mark.parametrize(
    ("links", "expected"),
    [
        (pytest.example_string_links, pytest.example_string_links),
        (pytest.example_links, pytest.example_links),
        ({}, {}),
        (None, TypeError),
    ],
)
def test_setLinkIn(init_actor, example_string_links, example_links, links, expected):
    """Tests if we can set the input queue."""

    act = init_actor
    act.setLinks(links)
    if links is not None:
        act.setLinkIn("input_q")
        expected.update({"q_in": "input_q"})
        assert act.links == expected
    else:
        with pytest.raises(AttributeError):
            act.setLinkIn("input_queue")


@pytest.mark.parametrize(
    ("links", "expected"),
    [
        (pytest.example_string_links, pytest.example_string_links),
        (pytest.example_links, pytest.example_links),
        ({}, {}),
        (None, TypeError),
    ],
)
def test_setLinkOut(init_actor, example_string_links, example_links, links, expected):
    """Tests if we can set the output queue."""

    act = init_actor
    act.setLinks(links)
    if links is not None:
        act.setLinkOut("output_q")
        expected.update({"q_out": "output_q"})
        assert act.links == expected
    else:
        with pytest.raises(AttributeError):
            act.setLinkIn("output_queue")


@pytest.mark.parametrize(
    ("links", "expected"),
    [
        (pytest.example_string_links, pytest.example_string_links),
        (pytest.example_links, pytest.example_links),
        ({}, {}),
        (None, TypeError),
    ],
)
def test_setLinkWatch(init_actor, example_string_links, example_links, links, expected):
    """Tests if we can set the watch queue."""

    act = init_actor
    act.setLinks(links)
    if links is not None:
        act.setLinkWatch("watch_q")
        expected.update({"q_watchout": "watch_q"})
        assert act.links == expected
    else:
        with pytest.raises(AttributeError):
            act.setLinkIn("input_queue")


def test_addLink(setup_store):
    """Tests if a link can be added to the dictionary of links."""

    act = Actor("test")
    links = {"1": "one", "2": "two"}
    act.setLinks(links)
    newName = "3"
    newLink = "three"
    act.addLink(newName, newLink)
    links.update({"3": "three"})

    # trying to check for two separate conditions while being able to
    # distinguish between them should an error be raised
    passes = []
    err_messages = []

    if act.getLinks()["3"] == "three":
        passes.append(True)
    else:
        passes.append(False)
        err_messages.append(
            "Error:\
            actor.getLinks()['3'] is not equal to \"three\""
        )

    if act.getLinks() == links:
        passes.append(True)
    else:
        passes.append("False")
        err_messages.append(
            "Error:\
            actor.getLinks() is not equal to the links dictionary"
        )

    err_out = "\n".join(err_messages)
    assert all(passes), f"The following errors occurred: {err_out}"


def test_getLinks(init_actor, example_string_links):
    """Tests if we can access the dictionary of links.

    TODO:
        Add more parametrized test cases.
    """

    act = init_actor
    links = example_string_links
    act.setLinks(links)

    assert act.getLinks() == {"1": "one", "2": "two", "3": "three"}


@pytest.mark.skip(
    reason="this is something we'll do later because\
                    we will subclass actor w/ watcher later"
)
def test_put(init_actor):
    """Tests if data keys can be put to output links.

    TODO:
        Ask Anne to take a look.
    """

    act = init_actor
    act.put()
    assert True


def test_run(init_actor):
    """Tests if actor.run raises an error."""
    act = init_actor
    with pytest.raises(NotImplementedError):
        act.run()


def test_changePriority(init_actor):
    """Tests if we are able to change the priority of an actor."""

    act = init_actor
    act.lower_priority = True
    act.changePriority()

    assert psutil.Process(os.getpid()).nice() == 19


def test_actor_connection(setup_store):
    """Test if the links between actors are established correctly.

    This test instantiates two actors with different names, then instantiates
    a Link object linking the two actors. A string is put to the input queue of
    one actor. Then, in the other actor, it is removed from the queue, and
    checked to verify it matches the original message.
    """

    act1 = Actor("a1")
    act2 = Actor("a2")

    Store(store_loc="/tmp/store")
    link = Link("L12", act1, act2)
    act1.setLinkIn(link)
    act2.setLinkOut(link)

    msg = "message"

    act1.q_in.put(msg)

    assert act2.q_out.get() == msg

def test_zmq_ps(ip, port):
    """Tests if we can set the zmq socket and send message."""

    act1 = ZmqPSActor("act1")
    act2 = ZmqPSActor("act2")

    try:
        act1.setSendSocket(ip, port)
        print("act1 set send socket success to ", act1.address)
    except:
        traceback.print_exc()
    try:
        act2.setRecvSocket(ip, port)
        print("act2 set recv socket success to ", act2.address)
    except:
        traceback.print_exc()
    # msg = input("input msg:")
    msg = "hello"
    try:
        act1.sendMsg(msg)
        print("act1 send result:", msg)
    except:
        traceback.print_exc()
    try:
        recvmsg = act2.recvMsg()
        print("act2 recv result:", recvmsg)
    except:
        traceback.print_exc()

async def test_zmq_rr(ip, port):

    act1 = ZmqRRActor("act1")
    act2 = ZmqRRActor("act2")
    act1.setReqSocket(ip, port)
    print("act1 set req socket success to ", act1.address)
    act2.setRepSocket(ip, port)
    print("act2 set rep socket success to ", act2.address)
    msg = "hello"
    reply = "world"
    coro1 = act1.requestMsg(msg)
    coro2 = act2.replyMsg(reply)
    result = await asyncio.gather(coro1, coro2)
    replymsg = result[0]
    recvmsg = result[1]
    print("act1 send result:", msg, ", and act2 reply result:", replymsg)
    print("act2 recv result:", recvmsg)

# write a main function to run the zmq test
if __name__ == "__main__":
    test_zmq_ps(ip = "127.0.0.1", port = 5555)
    asyncio.run(test_zmq_rr(ip = "127.0.0.1", port = 5556))