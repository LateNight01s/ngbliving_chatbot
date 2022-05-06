import React, { useEffect } from "react";
import "./App.css";
import { Widget, addResponseMessage, addLinkSnippet } from "react-chat-widget";

import "react-chat-widget/lib/styles.css";
import logo from "./logo-AWS-1024x658.webp";

function App() {
  const link = {
    title: "Project Repository",
    link: "https://github.com/LateNight01s/ngbliving_chatbot",
    target: "_blank",
  };

  async function postData(url = "", payload = {}) {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    return response;
  }

  async function handleNewUserMessage(msg) {
    const newMsg = { message: msg };
    await postData("http://52.66.245.122:8080/", newMsg)
      .then((response) => response.text())
      .then((response) => {
        addResponseMessage(response);
        console.log(response);
      })
      .catch((error) => {
        console.log(error);
      });

    if (msg.includes("Project") || msg.includes("project")) {
      addLinkSnippet(link);
    } else if (msg.includes("image")) {
      addResponseMessage("![prototype](https://i.imgur.com/10C1C9o.png)");
    }
  }

  useEffect(() => {
    addResponseMessage("Welcome to the CCL ChatBot");
    return () => {};
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <p>Welcome to CCL ChatBot</p>
      </header>
      <Widget
        title="CCL ChatBot"
        subtitle="Practical"
        profileAvatar={logo}
        handleNewUserMessage={handleNewUserMessage}
      />
    </div>
  );
}

export default App;
