"""Account store: migration from the single-password file, roles, guards."""

import json

import pytest

from webapp import auth, config


@pytest.fixture(autouse=True)
def auth_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTH_FILE", tmp_path / "auth.json")
    yield tmp_path / "auth.json"


def test_fresh_install_has_one_admin_with_forced_change():
    auth.ensure_initialised()
    users = auth.list_users()
    assert [u["username"] for u in users] == [config.DEFAULT_USERNAME]
    assert users[0]["role"] == "admin" and users[0]["must_change"]
    assert "password_hash" not in users[0]
    assert auth.authenticate(config.DEFAULT_USERNAME, config.DEFAULT_PASSWORD)
    assert auth.authenticate(config.DEFAULT_USERNAME, "wrong") is None
    assert auth.authenticate("nobody", config.DEFAULT_PASSWORD) is None


def test_single_password_file_migrates_to_admin_account(auth_file):
    """The pre-accounts data dir: one bare password hash. Its password must
    keep working, as the `admin` user, without a forced change."""
    auth_file.write_text(json.dumps({
        "password_hash": auth._hash_password("old-secret-1"),
        "must_change": False, "is_default": False,
        "updated_at": "2026-01-01T00:00:00+00:00",
    }))
    auth.ensure_initialised()
    doc = json.loads(auth_file.read_text())
    assert doc["version"] == 2 and len(doc["users"]) == 1
    u = auth.authenticate("admin", "old-secret-1")
    assert u and u["role"] == "admin" and u["id"] == 1 and not u["must_change"]
    assert auth.primary_admin_id() == 1
    # idempotent — a second start doesn't touch it
    auth.ensure_initialised()
    assert json.loads(auth_file.read_text())["users"][0]["password_hash"] == doc["users"][0]["password_hash"]


def test_create_user_validation_and_roles():
    auth.ensure_initialised()
    bob = auth.create_user("Bob", "bobs-password", role="user", must_change=False)
    assert bob["username"] == "bob" and bob["id"] == 2 and bob["role"] == "user"
    with pytest.raises(ValueError):
        auth.create_user("bob", "another-password")          # duplicate
    with pytest.raises(ValueError):
        auth.create_user("bad name!", "another-password")    # bad username
    with pytest.raises(ValueError):
        auth.create_user("carol", "short")                   # too short
    with pytest.raises(ValueError):
        auth.create_user("carol", "long-enough", role="root")
    assert not auth.is_admin(bob)
    auth.set_role(bob["id"], "admin")
    assert auth.is_admin(auth.get_user(bob["id"]))
    assert sorted(auth.admin_ids()) == [1, 2]


def test_last_enabled_admin_is_protected():
    auth.ensure_initialised()
    admin = auth.get_user_by_username(config.DEFAULT_USERNAME)
    with pytest.raises(ValueError):
        auth.set_role(admin["id"], "user")
    with pytest.raises(ValueError):
        auth.set_disabled(admin["id"], True)
    with pytest.raises(ValueError):
        auth.delete_user(admin["id"])
    other = auth.create_user("second", "second-password", role="admin")
    auth.set_role(admin["id"], "user")                # fine now — `second` remains
    with pytest.raises(ValueError):
        auth.delete_user(other["id"])                 # ...but `second` is now the last


def test_disable_and_reset_invalidate_sessions():
    auth.ensure_initialised()
    bob = auth.create_user("bob", "bobs-password", must_change=False)
    sv0 = bob["session_version"]

    auth.set_disabled(bob["id"], True)
    assert auth.authenticate("bob", "bobs-password") is None
    assert auth.get_user(bob["id"])["session_version"] > sv0
    auth.set_disabled(bob["id"], False)
    assert auth.authenticate("bob", "bobs-password")

    sv1 = auth.get_user(bob["id"])["session_version"]
    temp = auth.reset_password(bob["id"])
    u = auth.get_user(bob["id"])
    assert u["must_change"] and u["session_version"] > sv1
    assert auth.authenticate("bob", "bobs-password") is None
    assert auth.authenticate("bob", temp)
    assert auth.check_password(bob["id"], temp)

    auth.set_password(bob["id"], "brand-new-password")
    assert not auth.get_user(bob["id"])["must_change"]
    assert auth.authenticate("bob", "brand-new-password")


def test_delete_user_removes_account():
    auth.ensure_initialised()
    bob = auth.create_user("bob", "bobs-password")
    auth.delete_user(bob["id"])
    assert auth.get_user(bob["id"]) is None
    assert auth.get_user_by_username("bob") is None
    # ids are never reused
    assert auth.create_user("bob", "bobs-password")["id"] == 3


def test_cli_reset_admin(capsys):
    auth.ensure_initialised()
    admin = auth.get_user_by_username(config.DEFAULT_USERNAME)
    auth.set_password(admin["id"], "changed-password")
    assert auth._cli(["reset-admin"]) == 0
    assert auth.authenticate(config.DEFAULT_USERNAME, config.DEFAULT_PASSWORD)["must_change"]
    assert auth._cli(["create", "dave", "--admin"]) == 0
    assert "temporary password" in capsys.readouterr().out
    assert auth.get_user_by_username("dave")["role"] == "admin"
