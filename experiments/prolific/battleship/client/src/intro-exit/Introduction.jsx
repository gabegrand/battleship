import React from "react";
import { InstructionsButton } from "../components/InstructionsButton";
import { usePlayer, useGame } from "@empirica/core/player/classic/react";
import "./Introduction.css";

export function Introduction({ next }) {

  const player = usePlayer();
  const game = useGame();
  const treatment = game.get("treatment");

  return (
    <div className="mt-3 sm:mt-5" style={{display:"flex", flexDirection:"column", alignItems:"center"}}>
      <h3 className="text-lg leading-6 font-medium text-gray-900" style={{fontSize:"2.75vh", margin: "3vh"}}>
        <u>Instructions</u>
      </h3>
      <div style={{display:"flex", flexDirection:"column", margin: "2vh", gap:"2vh", width:"50vw"}}>
      <p style={{fontSize:"3vh"}}>Welcome to <i>Collaborative Battleship</i>!</p>
      <p>In this study, you will play a collaborative version of the board game Battleship with another Prolific participant. As a team, your goal is to sink all of the hidden ships by revealing their tiles on the board. You and your partner will alternate between two roles:</p>
      <ul>
        <li>As <b>Captain</b>, you will ask your partner questions about the board to try to get information about the hidden ships. Every turn, the Captain decides where to fire.</li>
        <li>As <b>Spotter</b>, you will answer questions for your partner, given special knowledge of the ship locations.</li>
      </ul>
      <p>Your bonus will be determined by how well you work together as a team: the fewer moves it takes for you to sink all the ships, the better. To maximize your team's bonus, the Captain will need to think carefully about what questions will be most informative, and the Spotter will need to answer the Captain's questions correctly. While there is a time limit for each turn, there is no bonus for playing quickly, so take your time to think through each move.</p>
      <p>First, we'll take you through a short tutorial to familiarize you with the interface. Then you will play <b>{treatment.testNumber}</b> training games on a small board, followed by <b>{treatment.realNumber}</b> test games on a larger one.</p>
      <p>Press Next to proceed to the tutorial.</p>
      </div>
      <InstructionsButton handleClick={next}>
        <p>Next</p>
      </InstructionsButton>

    </div>
  );
}
