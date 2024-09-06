import React, { useEffect, useState } from 'react';
import { Button } from "./Button.jsx";
import { usePlayer, useGame } from '@empirica/core/player/classic/react';
import { NextGameButton } from './NextGameButton.jsx';

export function InstructionBox(textStage, onClick) {
  const player = usePlayer();
  const game = useGame();
  
  function handleQuestionAdvice(){
    const categorical = game.get("categoricalAnswers");
    const limitedQuestions = !game.get("questionEveryTime");
    if (categorical && limitedQuestions) {
      return <p>However, note that you can only ask a limited number of questions and the spotter can only answer with <b>Yes</b> or <b>No</b>.</p>;
    }
    if (categorical && !limitedQuestions) {
      return <p>However, note that the spotter can only answer with <b>Yes</b> or <b>No</b>.</p>;
    }
    if (!categorical && limitedQuestions) {
      return <p>However, note that you can only ask a limited number of questions.</p>;
    }
    if (!categorical && !limitedQuestions) {
      return <p></p>;
    }

    document.body.innerHTML += text;

    return text
  }
  
  function handleText(stage) {
    switch (stage) {
      case 1:
        return (<p>Welcome to the Tutorial! You will start in the role of <b>Captain</b>. The board is below: tiles you haven't fired at yet are marked in light gray. Click continue to advance. </p>);
      case 2:
        return (<p>The colored tiles are ships. To sink a ship, you must reveal all its hidden tiles. Your goal is to sink all the ships on the board. </p>);
      case 3:
        return (<p>To the left of the board you have your <b>ship tracker</b>. The ship tracker tells you what color ships are on the board, and how long each ship is. Ships can vary in length from 2-5 tiles. When you sink a ship, its icon on the ship tracker lights up in its color: here, the green ship has been sunk, and we can see that it was 4 tiles long.</p>);
      case 4:
        return (<p>The light blue tiles are <b>water tiles</b>. Water is revealed when your shots miss a ship. Each miss reduces your bonus, so make sure to choose where to fire carefully.</p>);
      case 5:
        return (<p>To help you decide where to fire, you can ask questions about the board to the <b>Spotter</b>.{handleQuestionAdvice()}Let's try asking a question! Select the option to ask a question and type, "Is the red ship horizontal?"</p>);
      case 6:
        return (<p>Looks like the Spotter got back to you! Notice the "chat history" window on the right of the board: this archives all previous questions and answers so you can come back to them during the game. <br/>Fire at a tile to advance.</p>);
      case 7:
        return (<p>Well done! You will now play as the <b>Spotter</b>. Click continue to advance.</p>);
      case 8:
        return (<p>This is the Spotter's view: the Spotter can see the entire board, with tiles the captain cannot see marked with a <b>diagonal pattern</b>.</p>);
      case 9:
        return (<p>The Spotter's role is to answer the Captain's questions accurately: to end the tutorial, answer the question the Captain asked. It should look pretty familiar...</p>);
      case 10:
        return (<p>Good job! This concludes the Tutorial. Press the button below to continue to your first game!</p>);
    }
  }

  function handleEnd(){
    player.stage.set("timedOut",false);
    player.stage.set("submit",true);
  }

  function chooseButtons(stage) {
    if (stage < 10) {
      return (<div style={{display:"flex", flexDirection:"row", gap:"0.3vw"}}>
      {!([1,5,6,7,8,9].includes(player.stage.get("textStage"))) ? <Button height="2vw" width="6vw" fontSize="0.9vw" handleClick={() => player.stage.set("textStage",player.stage.get("textStage")-1)}>Back</Button> : <div></div>}
      {(!([5,6,9].includes(player.stage.get("textStage"))) || (player.stage.get("textStage") == 7 && player.stage.get("forcedUpdate") == undefined)) ? <Button height="2vw" width="6vw" fontSize="0.9vw" handleClick={() => player.stage.set("textStage",player.stage.get("textStage")+1)}>Continue</Button> : <div></div>}
      </div> );
    } else {
      return (<NextGameButton handleClick={handleEnd()}>Continue</NextGameButton>);
    }
  }

  return (<div style={{display:"flex", flexDirection:"column", paddingTop:"2vh", paddingLeft:"1.2vw", paddingRight:"1.2vw", paddingBottom:"2vh", gap:"2vh", borderRadius:"0.4vw", color:"black", backgroundColor: "rgba(51, 170, 51, .2)", border: "2px solid rgba(51, 170, 51)", alignItems:"center", textAlign:"center"}}>
    <div style={{width:"35vw", height:"auto", fontSize:"1vw"}}>
    {handleText(player.stage.get("textStage"))}
    </div>
    {chooseButtons(player.stage.get("textStage"))}
  </div>);
}

export default InstructionBox;
