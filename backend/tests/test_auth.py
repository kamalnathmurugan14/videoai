def test_signup_login_me(client):
    r = client.post("/api/auth/signup", json={"email": "Alice@Example.com", "password": "supersecret"})
    assert r.status_code == 201
    assert r.json()["user"]["email"] == "alice@example.com"
    assert client.post("/api/auth/signup", json={"email": "alice@example.com", "password": "supersecret"}).status_code == 409
    r = client.post("/api/auth/login", json={"email": "alice@example.com", "password": "supersecret"})
    assert r.status_code == 200
    token = r.json()["access_token"]
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200 and me.json()["is_guest"] is False


def test_bad_credentials_and_validation(client):
    assert client.post("/api/auth/login", json={"email": "nobody@example.com", "password": "whatever123"}).status_code == 401
    assert client.post("/api/auth/signup", json={"email": "not-an-email", "password": "supersecret"}).status_code == 422
    assert client.post("/api/auth/signup", json={"email": "b@example.com", "password": "short"}).status_code == 422


def test_guest_mode(client):
    r = client.post("/api/auth/guest")
    assert r.status_code == 201 and r.json()["user"]["is_guest"] is True
    h = {"Authorization": f"Bearer {r.json()['access_token']}"}
    assert client.get("/api/projects", headers=h).json() == []


def test_requires_auth(client):
    assert client.get("/api/projects").status_code == 401
    assert client.get("/api/projects", headers={"Authorization": "Bearer garbage"}).status_code == 401
